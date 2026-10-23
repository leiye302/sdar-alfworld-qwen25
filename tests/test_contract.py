import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import checkpoint_complete, devices, load, protocol, tasks
from fixed_validation import fixed_validation_envs
from run import config
from setup import verify


def test_native_source_and_task_locks():
    lock = verify()
    assert len(lock['unchanged_files']) >= 10
    assert 'verl/workers/actor/dp_actor.py' in lock['unchanged_files']
    assert len(tasks()) == 140


@pytest.mark.parametrize('size', ['1.5b', '3b'])
def test_composed_native_recipe(size, tmp_path):
    cfg = config(size, tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    assert (cfg.trainer.n_gpus_per_node, cfg.ray_init.num_gpus) == (4, 4)
    assert cfg.algorithm.sdar.gate_beta == 5 and cfg.algorithm.sdar.sdar_coef == 0.01
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 256
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.01
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0.001
    assert cfg.env.max_steps == 50 and cfg.env.history_length == 2
    assert cfg.data.max_response_length == 512
    assert cfg.data.val_batch_size == 140
    assert cfg.trainer.total_training_steps == 150
    assert cfg.trainer.test_freq == 10 and cfg.trainer.save_freq == 20
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    assert 'local_hints' not in cfg.algorithm
    assert cfg.actor_rollout_ref.model.get('lora_rank', 0) == 0


@pytest.mark.parametrize('spec', ['0,1', '0,1,2,2', '0,1,2,3,4', '', '0,1,2,GPU-xxx'])
def test_invalid_gpu_allocation(spec):
    with pytest.raises(ValueError):
        devices(spec)


def test_four_gpu_allocation():
    assert devices('4,5,6,7') == ['4', '5', '6', '7']


def test_partial_checkpoint_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        checkpoint_complete(tmp_path)
    for rank in range(4):
        for kind in ('model', 'optim', 'extra_state'):
            path = tmp_path / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'unit-test')
    (tmp_path/'actor/config.json').write_text('{}')
    (tmp_path/'data.pt').write_bytes(b'unit-test')
    assert checkpoint_complete(tmp_path) == tmp_path


def test_telemetry_does_not_accept_duplicate_actions():
    assert protocol('<think>Plan.</think><action>look</action>')['strict_protocol']
    assert not protocol('<think></think><action>look</action>')['strict_protocol']
    assert not protocol('<think>Plan.</think><action>look</action><action>look</action>')['strict_protocol']


def test_fixed_task_uses_only_seen_and_preserves_base(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    game = tmp_path/'json_2.1.1/valid_seen/task/trial/game.tw-pddl'
    game.parent.mkdir(parents=True)
    game.write_bytes(b'game')
    manifest = tmp_path/'manifest.json'
    row = {'task_id': game.relative_to(tmp_path).as_posix(), 'seed': 2026,
           'game_sha256': hashlib.sha256(b'game').hexdigest()}
    manifest.write_text(json.dumps([row]))
    base = SimpleNamespace(game_files=[str(game)], num_games=140)
    selected = fixed_validation_envs(base, manifest, 1, 1)
    assert selected[0].fixed_validation_seed == 2026
    assert base.num_games == 140 and selected[0] is not base
    row['game_sha256'] = '0'*64
    manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        fixed_validation_envs(base, manifest, 1, 1)


def test_native_sdar_gradient_is_teacher_dependent_and_teacher_detached():
    torch = pytest.importorskip('torch')
    source = ROOT/'.deps/sdar/verl/trainer/ppo/sdar_utils.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_sdar_loss')
    namespace = {'torch': torch, 'agg_loss': lambda loss_mat, loss_mask, loss_agg_mode:
                 (loss_mat*loss_mask).sum()/loss_mask.sum()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    gradients = []
    for teacher_value in (-0.1, -3.0):
        student = torch.tensor([[-1.0, -1.0]], requires_grad=True)
        teacher = torch.full_like(student, teacher_value, requires_grad=True)
        mask = torch.tensor([[1.0, 0.0]])
        loss, _ = namespace['compute_sdar_loss'](student, teacher, mask, gate_beta=5.0)
        loss.backward()
        assert teacher.grad is None and student.grad[0, 1].item() == 0
        gradients.append(student.grad[0, 0].item())
    assert gradients[0] < -0.9 and abs(gradients[1]) < 0.001


def test_worker_pool_preserves_native_seeds_order_and_results():
    source = ROOT/'.deps/sdar/agent_system/environments/env_package/alfworld/envs.py'
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {'AlfworldWorker', 'AlfworldWorkerPool', 'AlfworldEnvs'}]

    class NativeEnvironment:
        def seed(self, seed): self.value = seed
        def reset(self): return [str(self.value)], {'admissible_commands': [['look']]}
        def step(self, actions):
            return [str(self.value)+actions[0]], [0], [False], {'admissible_commands': [['look']], 'won': [False]}

    class Base:
        def init_env(self, batch_size): return NativeEnvironment()

    class Method:
        def __init__(self, fn): self.remote = fn

    class Handle:
        def __init__(self, obj): self.obj = obj
        def __getattr__(self, name): return Method(getattr(self.obj, name))

    class Remote:
        def __init__(self, cls): self.cls = cls
        def remote(self, *args): return Handle(self.cls(*args))

    ray = SimpleNamespace(is_initialized=lambda: True, get=lambda x:x,
                          remote=lambda **kwargs: lambda cls:Remote(cls))
    namespace = {'gym': SimpleNamespace(Env=object), 'ray': ray, 'get_environment': lambda _: lambda *a,**k:Base(),
                 'load_config_file': lambda _: {'env': {'type': 'AlfredTWEnv'}},
                 'compute_reward': lambda info,multi_modal: 0}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    outputs = []
    for pool_size in (1, 2, 4):
        env = namespace['AlfworldEnvs']('unused', 17, 3, 2, {}, env_kwargs={'worker_pool_size':pool_size})
        reset = copy.deepcopy(env.reset())
        step = copy.deepcopy(env.step(['look']*6))
        outputs.append((reset, step))
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][0][0] == ['17', '17', '18', '18', '19', '19']


def test_dry_run_needs_no_model_or_gpu(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'src/run.py'), 'train', '--size', '1.5b',
                             '--run-dir', str(tmp_path/'unused'), '--dry-run'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'unused').exists()
    assert 'gate_beta: 5.0' in result.stdout


def test_models_share_all_scientific_settings(tmp_path):
    from omegaconf import OmegaConf
    left = config('1.5b', tmp_path/'model15', tmp_path/'assets', tmp_path/'run15')
    right = config('3b', tmp_path/'model3', tmp_path/'assets', tmp_path/'run3')
    for cfg in (left, right):
        cfg.actor_rollout_ref.model.path = 'model'
        cfg.critic.model.tokenizer_path = 'model'
        cfg.reward_model.model.input_tokenizer = 'model'
        cfg.trainer.experiment_name = 'experiment'
        cfg.data.train_files = cfg.data.val_files = 'parquet'
        cfg.trainer.default_local_dir = cfg.trainer.validation_data_dir = cfg.trainer.rollout_data_dir = 'output'
        cfg.ray_init.runtime_env.env_vars.SDAR_RUN_DIR = 'output'
    assert OmegaConf.to_container(left, resolve=True) == OmegaConf.to_container(right, resolve=True)


def test_evaluation_hook_preserves_native_success_and_all_tasks(tmp_path, monkeypatch):
    import importlib.util
    import types
    torch = pytest.importorskip('torch')
    rows = tasks()
    gamefiles = [str(tmp_path/row['task_id']) for row in rows]
    batch = SimpleNamespace(
        batch={'responses': torch.ones((140, 2), dtype=torch.long),
               'attention_mask': torch.ones((140, 4), dtype=torch.long)},
        non_tensor_batch={'gamefile': gamefiles, 'traj_uid': [str(i) for i in range(140)],
                          'turn_step': [0]*140, 'episode_rewards': [10.0]*70+[0.0]*70,
                          'is_action_valid': [True]*140})

    class NativeTrainer:
        def _validate(self):
            self.traj_collector.multi_turn_loop()
            return {'val/success_rate': 0.5}

    native_module = types.ModuleType('verl.trainer.ppo.skillsd_ray_trainer')
    native_module.SkillSDRayTrainer = NativeTrainer
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.skillsd_ray_trainer', native_module)
    spec = importlib.util.spec_from_file_location('tested_sdar_eval', ROOT/'src/sdar_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    monkeypatch.setenv('SDAR_RUN_DIR', str(tmp_path/'run'))
    trainer = module.FixedSeenTrainer()
    trainer.global_steps = 10
    original = lambda:batch
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=original)
    trainer.tokenizer = SimpleNamespace(decode=lambda ids,**kw:'<think>Plan.</think><action>look</action>')
    assert trainer._validate() == {'val/success_rate':0.5}
    assert trainer.traj_collector.multi_turn_loop is original
    summary = json.loads((tmp_path/'run/evaluation/update_010/summary.json').read_text())
    assert summary['successes'] == 70 and len(summary['task_results']) == 140
    assert summary['reasoning_complete_rate'] == 1
