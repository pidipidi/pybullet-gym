from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.custom.robots.peg.gripper_2d import Gripper2D
from pybulletgym.envs.roboschool.scenes.peg_in_shallow_hole import PegInShallowHoleScene
import numpy as np


RENDER_HEIGHT = 240
RENDER_WIDTH = 320

class GripperPegInHole2DPyBulletEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = Gripper2D()
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
        ## self.render(mode='human')
        self._cam_dist = 0.4
        self._cam_yaw = 0
        self._cam_pitch = -90
        self._render_width = RENDER_WIDTH
        self._render_height = RENDER_HEIGHT
        
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space

    def create_single_player_scene(self, bullet_client):
        #self.robot.jdict['x_slider'].reset_current_position(1. , 0.)

        # peg in shallow hole scene
        timestep = 1./8000. #1./240.
        scene = PegInShallowHoleScene(bullet_client, gravity=[0, -9.8, 0], timestep=timestep, frame_skip=1)
        scene.episode_restart(self._p)
        self.robot.parts, self.robot.jdict, self.robot.ordered_joints, self.robot.robot_body = self.robot.addToScene(self._p, scene.stadium)
        self.robot.parts, self.robot.jdict, self.robot.ordered_joints, self.robot.robot_body = self.robot.addToScene(self._p, scene.peg)
        
        return scene
        
        ## return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        r = BaseBulletEnv._reset(self)
        return r

    def step(self, a):
        ## assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()
        
        # x,xd,y,yd,theta,thetad,to_target_vec
        state = self.robot.calc_state()
        done  = self.terminate()

        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        # work velocity_input*angular_velocity
        electricity_cost = -0.001 * np.sum(np.square(a))

        # release reward
        release_rwd = self.robot.release_reward()

        # stuck cost
        stuck_joint_cost = -0.1 if np.abs(self.robot.y - self.robot.target.pose().xyz()[1]) < 0.02 else 0.

        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(release_rwd),
                        float(stuck_joint_cost)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def terminate(self):
        # if peg is not hold by the gripper
        ## p = self.robot.peg.pose().xyz()
        p = self.robot.peg_handle.pose().xyz()
        v = np.array([self.robot.x-p[0], self.robot.y-p[1]])
        d_gripper_peg = np.linalg.norm(v)
        if d_gripper_peg > 0.03:            
            ## print ("DONE: d_gripper_peg {}".format(d_gripper_peg))
            return True

        ang = abs(self.robot.theta-self.robot.peg.pose().rpy()[2])
        ang = ang if ang < np.pi else abs(ang-2*np.pi)
        if ang > np.pi/180. * 100. :
            ## print ("DONE: theta {}".format(ang/np.pi*180.))
            return True

        p = self.robot.peg.pose().xyz()
        if p[1] < self.robot.target.pose().xyz()[1]-0.005:
            ## print ("DONE: gripper is under the target")
            return True
        return False

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, -0.05, 0.5)




class GripperPegInHole2DPyBulletEnv_Force(GripperPegInHole2DPyBulletEnv):
    def __init__(self):
        self.robot = Gripper2D(add_feature='force')
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

        self._cam_dist = 0.4
        self._cam_yaw = 0
        self._cam_pitch = -90
        self._render_width = RENDER_WIDTH
        self._render_height = RENDER_HEIGHT
        
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space


class GripperPegInHole2DPyBulletEnv_ForceWindow(GripperPegInHole2DPyBulletEnv):
    def __init__(self):
        self.robot = Gripper2D(add_feature='force_window',
                               window_size=5)
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

        self._cam_dist = 0.5
        self._cam_yaw = 0
        self._cam_pitch = -90
        self._render_width = RENDER_WIDTH
        self._render_height = RENDER_HEIGHT
        
        self.action_space = self.robot.action_space
        self.observation_space = self.robot.observation_space
