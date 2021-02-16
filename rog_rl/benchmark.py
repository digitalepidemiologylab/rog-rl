#! ~/anaconda3/envs/epirl/bin/python

import time
from rog_rl import BaseGridRogRLEnv
import cProfile
import pstats

'''
Benchmark env time
Code adapted from
https://github.com/maximecb/gym-miniworld/blob/master/benchmark.py
'''
render_profile_file = 'profile_stats_render'


def performance_metrics(render_on=False):

    st = time.time()
    env_config = dict(use_renderer="simple", use_np_model=True)

    '''
    To compare against standard envs like cartpole
    '''
    # import gym
    # env = gym.make('CartPole-v0')

    env = BaseGridRogRLEnv(env_config)
    env.reset()
    load_time = 1000 * (time.time() - st)

    # Benchmark the reset time
    st = time.time()
    for i in range(3):
        env.reset()
    reset_time = 1000 * (time.time() - st) / 3

    # Benchmark the rendering/update speed
    st = time.time()
    for i in range(1):
        # Profile the code
        if render_on:
            cProfile.runctx('env.render()', globals(), locals(),
                            filename=render_profile_file)
        else:
            env.render()
    render_time = 1000_000 * (time.time() - st) / 1

    num_frames = 0
    st = time.time()

    while True:
        dt = time.time() - st

        if dt > 1:
            break

        # Slow movement speed to minimize resets
        obs, reward, done, info = env.step(env.action_space.sample())

        if done:
            env.reset()

        num_frames += 1

    fps = num_frames / dt
    frame_time = 1000 * dt / num_frames

    print()
    print('load time: {:,.1f} ms'.format(int(load_time)))
    print('reset time: {:,.1f} ms'.format(reset_time))
    print('render time: {:,.1f} ms'.format(render_time))  # \u03BCs'
    print('frame time: {:,.1f} ms'.format(frame_time))
    print('frame rate: {:,.1f} FPS'.format(fps))

    env.close()

    if render_on:
        profile(render_profile_file)


'''
View profile stats by Cumulative Time

'''


def profile(filename):

    p = pstats.Stats(filename)
    # p.strip_dirs().sort_stats(-1).print_stats()
    # p.print_stats()
    p.sort_stats('cumulative').print_stats(10)


if __name__ == "__main__":
    performance_metrics()
