import os.path
import os.path as osp
import argparse

variant = dict(
    layer_size=256,
    replay_buffer_size=int(1E6),
    algorithm_kwargs=dict(
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=None,
        num_expl_steps_per_train_loop=None,
        min_num_steps_before_training=10000,
        max_path_length=1000,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,

    ),
    exploration_kwargs=dict(
        sample_size=32,
    )
)


def get_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='oac')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--domain', type=str, default='invertedpendulum')
    parser.add_argument('--use_gpu', default=False, action='store_true')
    parser.add_argument('--no_aet', default=True, action='store_false')
    parser.add_argument('--task', type=str, default='performence')
    parser.add_argument('--sample_size', type=int, default=32)
    parser.add_argument('--sample_range', type=int, default=7, help='sample range')
    parser.add_argument('--beta', type=float, default=1.0, help='beta t')
    parser.add_argument('--log_model', default=False, action='store_true', help='beta t')

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=1000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)

    args = parser.parse_args()

    return args

def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):

    log_dir = args.algo

    if should_include_domain:
        log_dir = osp.join(log_dir, args.domain)

    if should_include_seed:
        log_dir = osp.join(log_dir, f'seed_{args.seed}')

    if should_include_base_log_dir:
        log_dir = osp.join(args.task, log_dir)

    # add home
    log_dir= osp.join(os.path.expanduser("~"), 'experiments',log_dir)
    if osp.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)

    return log_dir