from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.custom.robots.peg.cuboid_thin_peg_2d import CuboidPeg2D
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np



class CuboidPegInShallowHole2DPyBulletEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = CuboidPeg2D()
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
        ## self.render(mode='human')

    def create_single_player_scene(self, bullet_client):
        # plane_stadium_scene
        ## return self.plane_stadium_scene
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        ## if self.stateId >= 0:
        ##     print("CuboidPegBulletEnv reset p.restoreState(",self.stateId,")")
        ##     self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)

        # for manipulation arena
        if self.robot.doneLoading == 0:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self.robot._p, self.robot.arena)
        
        ## if self.stateId < 0:
        ##     self.stateId = self._p.saveState()
        ## print("CuboidPegBulletEnv reset self.stateId=",self.stateId)
        return r

    def step(self, a):
        ## assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()
        
        # x,xd,y,yd,theta,thetad,to_target_vec
        state = self.robot.calc_state()

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

