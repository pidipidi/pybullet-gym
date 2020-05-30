from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.custom.robots.peg.gripper_2d import Gripper2D
from pybulletgym.envs.roboschool.scenes.peg_in_shallow_hole import PegInShallowHoleScene
import numpy as np
import pybullet
import gym
import cv2
import matplotlib.pyplot as plt

RENDER_HEIGHT = 240
RENDER_WIDTH = 320

CNN_IMG_HEIGHT = 64
CNN_IMG_WIDTH = 64

class GripperCamPegInHole2DPyBulletEnv(BaseBulletEnv):
    
    def __init__(self):
        self.robot = Gripper2D()
        self.initialize()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(1, CNN_IMG_HEIGHT,
                                                       CNN_IMG_WIDTH),
                                                dtype=np.float32)
        
    def initialize(self):
        BaseBulletEnv.__init__(self, self.robot)       
        self.stateId = -1
        ## self.render(mode='human')

        self._cam_dist = 0.4
        self._cam_yaw = 0
        self._cam_pitch = -90
        self._render_width = RENDER_WIDTH
        self._render_height = RENDER_HEIGHT
        self.robot.body_xyz = [0,-0.03,0]
        
        self.action_space = self.robot.action_space

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
        return self.observation()


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
        
        state = self.observation()
        ## from IPython import embed; embed(); sys.exit()
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def terminate(self):
        # if peg is not hold by the gripper
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

    def observation(self):
        rgb_array = self.render(mode="rgb_array")
        img = ProcessFrame.process(rgb_array)
        img = np.moveaxis(img, 2, 0) / 255.0
        return img


class GripperCamPegInHole2DPyBulletEnv_Force(GripperCamPegInHole2DPyBulletEnv):
    def __init__(self):
        self.robot = Gripper2D(add_feature='force')
        ## from IPython import embed; embed(); sys.exit()
        GripperCamPegInHole2DPyBulletEnv.initialize(self)
        
        self.observation_space = \
          gym.spaces.Box(low=0.0, high=1.0,
                         shape=(CNN_IMG_HEIGHT*CNN_IMG_WIDTH + \
                                self.robot.obs_dim-3,),
                         dtype=np.float32)
        
    def reset(self):
        f = BaseBulletEnv._reset(self)[:-3]
        state = self.observation()
        state = np.reshape(state, CNN_IMG_WIDTH*CNN_IMG_HEIGHT)
        return np.concatenate((state, f))        


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

        f = state[:-3]
        state = self.observation()
        state = np.reshape(state, CNN_IMG_WIDTH*CNN_IMG_HEIGHT)
        state = np.concatenate((state, f))        
        ## from IPython import embed; embed(); sys.exit()
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}



class GripperCamPegInHole2DPyBulletEnv_ForceWindow(GripperCamPegInHole2DPyBulletEnv):
    def __init__(self):
        self.window_size = 5
        self.robot = Gripper2D(add_feature='force_window',
                               window_size=self.window_size)
        GripperCamPegInHole2DPyBulletEnv.initialize(self)
        self.observation_space = \
          gym.spaces.Box(low=0.0, high=1.0,
                         shape=(CNN_IMG_HEIGHT*CNN_IMG_WIDTH + \
                                self.robot.obs_dim-3,),
                         dtype=np.float32)
        
    def reset(self):
        f = BaseBulletEnv._reset(self)[:-3]
        state = self.observation()
        state = np.reshape(state, CNN_IMG_WIDTH*CNN_IMG_HEIGHT)
        return np.concatenate((state, f))        


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

        f = state[:-3]
        ## f = np.reshape(f, (self.window_size, -1))
        ## f -= f[0]
        ## f = f.flatten().tolist()
        
        state = self.observation()
        state = np.reshape(state, CNN_IMG_WIDTH*CNN_IMG_HEIGHT)
        state = np.concatenate((state, f))        
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}




class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(CNN_IMG_HEIGHT, CNN_IMG_WIDTH, 1), dtype=np.uint8)
        
    def observation(self, obs):
        return ProcessFrame.process(self.env.render(mode="rgb_array"))

    @staticmethod
    def process(frame):
        if frame.size == RENDER_WIDTH * RENDER_HEIGHT * 3:
            img = np.reshape(frame, [RENDER_HEIGHT,
                                     RENDER_WIDTH, 3]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        ## from IPython import embed; embed(); sys.exit()
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + \
          img[:, :, 2] * 0.11
        #img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
        #  img[:, :, 2] * 0.114
          
        resized_screen = cv2.resize(
            img, (int(CNN_IMG_HEIGHT*RENDER_WIDTH/float(RENDER_HEIGHT)),
                  CNN_IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        #from IPython import embed; embed(); sys.exit()
        ## cv2.imshow('image', img)
        ## cv2.waitKey(0)
        ## cv2.destroyAllWindows()
        I = int(len(resized_screen[0])/2-CNN_IMG_WIDTH/2)
        y_t = resized_screen[:, I:I+CNN_IMG_WIDTH]
        y_t = np.reshape(y_t, [CNN_IMG_HEIGHT, CNN_IMG_WIDTH, 1])

        ## y_t -= np.mean(y_t)
        y_t -= np.amin(y_t)
        y_t /= np.amax(y_t)
        y_t *= -255.
        y_t += 255.
        
        ## print (np.amax(y_t), np.amin(y_t))
        
        ## plt.imshow(y_t.squeeze(), cmap='gray')
        ## plt.show()
        return y_t.astype(np.uint8)
        
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    @staticmethod
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)/255.0
                                                        
