from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.custom.robots.peg.gripper_2d import Gripper2D
from pybulletgym.envs.roboschool.scenes.peg_in_shallow_hole import PegInShallowHoleScene
import numpy as np



class GripperPegInHole2DPyBulletEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = Gripper2D()
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
        ## self.render(mode='human')

    def create_single_player_scene(self, bullet_client):
        # peg in shallow hole scene
        timestep = 1./240.
        #timestep = 0.0165
        scene = PegInShallowHoleScene(bullet_client, gravity=9.8, timestep=timestep, frame_skip=1)
        scene.episode_restart(self._p)
        self.robot.parts, self.robot.jdict, self.robot.ordered_joints, self.robot.robot_body = self.robot.addToScene(self._p, scene.stadium)
        self.robot.parts, self.robot.jdict, self.robot.ordered_joints, self.robot.robot_body = self.robot.addToScene(self._p, scene.peg)
        return scene
        
        ## return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        r = BaseBulletEnv._reset(self)

        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
        ## from IPython import embed; embed(); sys.exit()

        return r

    def step(self, a):
        ## assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()
        
        # x,xd,y,yd,theta,thetad,to_target_vec
        state = self.robot.calc_state()

        # if peg is not hold by the gripper
        p = self.robot.peg.pose().xyz()
        d_gripper_peg = np.linalg.norm([p[0]-state[0], p[1]-state[2]])
        if d_gripper_peg > 0.1:
            done = True
        else:
            #done = abs(state[-3])<0.01 and abs(state[-2])<0.01 and abs(state[-1])<0.15
            done = False

        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        electricity_cost = (
            # work velocity_input*angular_velocity
            -0.001 * (np.abs(a[0] * state[1]) + np.abs(a[1] * state[3]) + np.abs(a[2] * state[5]))
        )

        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)

