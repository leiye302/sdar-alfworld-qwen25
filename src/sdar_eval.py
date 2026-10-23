"""Read-only metrics/evaluation hooks around the unchanged native SDAR trainer."""
import json
import os
from pathlib import Path
from common import atomic_json, protocol, tasks
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class FixedSeenTrainer(SkillSDRayTrainer):
    def fit(self):
        from verl.utils.tracking import Tracking
        original = Tracking.log
        destination = Path(os.environ['SDAR_RUN_DIR']) / 'metrics.jsonl'

        def capture(tracker, data, step, backend=None):
            original(tracker, data=data, step=step, backend=backend)
            record = {'step': int(step), 'metrics': data}
            with destination.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, default=lambda x: x.item(), allow_nan=False) + '\n')
                stream.flush()

        Tracking.log = capture
        try:
            return super().fit()
        finally:
            Tracking.log = original

    def _validate(self):
        original = self.traj_collector.multi_turn_loop
        records = []
        root = Path(os.environ['ALFWORLD_DATA']).resolve()
        expected = {row['task_id'] for row in tasks()}

        def capture(*args, **kwargs):
            batch = original(*args, **kwargs)
            ids = batch.batch['responses']
            mask = batch.batch['attention_mask'][:, -ids.shape[-1]:].bool()
            nt = batch.non_tensor_batch
            for i in range(len(ids)):
                tokens = ids[i][mask[i]].cpu().tolist()
                text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                game = Path(str(nt['gamefile'][i])).resolve().relative_to(root).as_posix()
                if game not in expected:
                    raise RuntimeError('Evaluation ran an unexpected task')
                records.append({'task_id': game, 'trajectory_id': str(nt['traj_uid'][i]),
                                'step': int(nt['turn_step'][i]), 'text': text, 'token_ids': tokens,
                                'episode_reward': float(nt['episode_rewards'][i]),
                                'action_valid': bool(nt['is_action_valid'][i]),
                                'protocol': protocol(text)})
            return batch

        self.traj_collector.multi_turn_loop = capture
        try:
            metrics = super()._validate()
        finally:
            self.traj_collector.multi_turn_loop = original
        by_task = {}
        for row in records:
            previous = by_task.setdefault(row['task_id'], row)
            if (previous['trajectory_id'], previous['episode_reward']) != (row['trajectory_id'], row['episode_reward']):
                raise RuntimeError('Duplicate task or inconsistent trajectory evaluation reward')
        if set(by_task) != expected:
            raise RuntimeError('Expected exactly one rollout for each seen140 task')
        successes = sum(row['episode_reward'] > 0 for row in by_task.values())
        summary = {'update': int(self.global_steps), 'split': 'valid_seen', 'tasks': 140,
                   'successes': successes, 'success_rate': successes / 140, 'decisions': len(records),
                   'reasoning_open_rate': sum(r['protocol']['reasoning_open'] for r in records) / len(records),
                   'reasoning_complete_rate': sum(r['protocol']['reasoning_nonempty_closed'] for r in records) / len(records),
                   'strict_protocol_rate': sum(r['protocol']['strict_protocol'] for r in records) / len(records),
                   'native_action_valid_rate': sum(r['action_valid'] for r in records) / len(records),
                   'native_metrics': metrics,
                   'task_results': [{'task_id': key, 'success': row['episode_reward'] > 0,
                                     'episode_reward': row['episode_reward']}
                                    for key, row in sorted(by_task.items())]}
        if abs(metrics['val/success_rate'] - summary['success_rate']) > 1e-6:
            raise RuntimeError('Task-level success differs from the native metric')
        out = Path(os.environ['SDAR_RUN_DIR']) / 'evaluation' / f'update_{self.global_steps:03d}'
        atomic_json(out / 'summary.json', summary)
        atomic_json(out / 'decisions.json', records)
        print(f'SEEN140 update={self.global_steps} successes={successes}/140 rate={successes / 140:.6f}')
        return metrics
