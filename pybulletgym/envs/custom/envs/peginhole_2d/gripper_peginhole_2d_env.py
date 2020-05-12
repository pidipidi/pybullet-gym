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
        #self.robot.jdict['x_slider'].reset_current_position(1. , 0.)

        # peg in shallow hole scene
        timestep = 1./1000. #1./240.
        scene = PegInShallowHoleScene(bullet_client, gravity=[0, 0, -9.8], timestep=timestep, frame_skip=1)
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
        #self.render(mode='human')

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

        self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(release_rwd)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def terminate(self):
        # if peg is not hold by the gripper
        p = self.robot.peg.pose().xyz()
        v = np.array([self.robot.x-p[0], self.robot.y-p[1]])
        d_gripper_peg = np.linalg.norm(v)
        if d_gripper_peg > 0.07:            
            done = True
        else:
            #done = abs(state[-3])<0.01 and abs(state[-2])<0.01 and abs(state[-1])<0.15
            done = False

        orn = self.robot.peg.pose().rpy()
        peg_v = np.array([np.cos(orn[2]), np.sin(orn[2])])
        #peg_n = np.array([np.cos(orn[2]+np.pi/2.), np.sin(orn[2]+np.pi/2.)])

        if done is False and np.sum(v/d_gripper_peg*peg_v) < -0.5:
            done = True
        return done

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, -0.05, 0.5)

