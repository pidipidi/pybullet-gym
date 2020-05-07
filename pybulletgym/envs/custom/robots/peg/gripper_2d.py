from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot
import numpy as np
import os
import pybullet_data, pybullet
import gym, gym.spaces, gym.utils

class Gripper2D(URDFBasedRobot):

    def __init__(self):
        URDFBasedRobot.__init__(self, '../robots/gripper/gripper_description/simple_gripper.urdf', 'cart', action_dim=4, obs_dim=11, basePosition=[0,0,0.025], fixed_base=True, self_collision=True)

    def robot_specific_reset(self, bullet_client):
        self._p     = bullet_client
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        #self._p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self._p.setTimeStep(1./240.)
        #self._p.setGravity(0, 0, -10)

        ## for i in range(self._p.getNumJoints(self.objects)):
        ##     print (self._p.getJointInfo(self.objects, i))
        #from IPython import embed; embed(); sys.exit()
        ## print (self._p.getDynamicsInfo(self.objects, 4))
        ## self._p.changeDynamics(self.objects, 4, contactDamping=_)
        ## import sys; sys.exit()
        
        self.target = self.parts["target"]
        self.peg    = self.parts["peg"]
        
        self.x_slider    = self.jdict["x_slider"]
        self.y_slider    = self.jdict["y_slider"]
        self.theta_joint = self.jdict["z_axis_joint"]
        self.left_finger_joint  = self.jdict["left_finger_joint"]
        self.right_finger_joint = self.jdict["right_finger_joint"]

        high  = [self.jdict['x_slider'].jointMaxVelocity,
                self.jdict['y_slider'].jointMaxVelocity,
                self.jdict['z_axis_joint'].jointMaxVelocity,
                self.jdict['left_finger_joint'].jointMaxVelocity]
        self.action_space = gym.spaces.Box(-np.array(high), np.array(high))

        high = [self.jdict['x_slider'].upperLimit,
                self.jdict['y_slider'].upperLimit,
                self.jdict['z_axis_joint'].upperLimit,
                self.jdict['left_finger_joint'].upperLimit]
        low = [self.jdict['x_slider'].lowerLimit,
               self.jdict['y_slider'].lowerLimit,
               self.jdict['z_axis_joint'].lowerLimit,
               self.jdict['left_finger_joint'].lowerLimit]
        self.observation_space.high[0] = high[0]
        self.observation_space.high[2] = high[1]
        self.observation_space.high[4] = high[2]
        self.observation_space.high[6] = high[3]
        self.observation_space.low[0] = low[0]
        self.observation_space.low[2] = low[1]
        self.observation_space.low[4] = low[2]
        self.observation_space.low[6] = low[3]


        # set pos wrt to the peg
        ## u = self.np_random.uniform(low=self.observation_space.low[0]*0.8,
        ##                            high=self.observation_space.high[0]*0.8)
        pos = self.peg.pose().xyz()
        rpy = self.peg.pose().rpy()
        self.theta_joint.reset_current_position( rpy[2] , 0)
        self.x_slider.reset_current_position( pos[0]+0.035*np.cos(rpy[2]) , 0)
        ## u = self.np_random.uniform(low=self.observation_space.low[2]*0.8,
        ##                            high=self.observation_space.high[2]*0.8)
        self.y_slider.reset_current_position( pos[1]+0.035*np.sin(rpy[2]) , 0)
        ## u = self.np_random.uniform(low=self.observation_space.low[4],
        ##                            high=self.observation_space.high[4])

        ## self.left_finger_joint.set_torque( -15 )
        ## self.right_finger_joint.set_torque( -15 )
        
        self.x_slider.set_velocity(0)
        self.y_slider.set_velocity(0)
        self.theta_joint.set_velocity(0)
        self.left_finger_joint.set_velocity(0)
        self.right_finger_joint.set_velocity(0)
        self.scene.global_step()
        
        p = 0.005
        self.left_finger_joint.set_position( p, force=3 )
        self.right_finger_joint.set_position( p, force=3 )
        ## self.left_finger_joint.reset_current_position( 0.005 , 0)
        ## self.right_finger_joint.reset_current_position( 0.005 , 0)
        
        self.scene.global_step()


    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0; a[1] = 0; a[2] = 0; a[3] = 0


        p = np.clip(a[0], self.action_space.low[0], self.action_space.high[0])
        p = float(np.clip(p+self.x,
                          self.observation_space.low[0],
                          self.observation_space.high[0]))
        self.x_slider.set_position( p , positionGain=0.1)

        p = np.clip(a[1], self.action_space.low[1], self.action_space.high[1])
        p = float(np.clip(p+self.y,
                          self.observation_space.low[2],
                          self.observation_space.high[2]))
        self.y_slider.set_position( p , positionGain=0.1)

        p = np.clip(a[2], self.action_space.low[2], self.action_space.high[2])
        p = float(np.clip(p+self.theta,
                          self.observation_space.low[4],
                          self.observation_space.high[4]))
        self.theta_joint.set_position( p , positionGain=0.1)

        p = float(np.clip(a[3],
                          self.observation_space.low[6],
                          self.observation_space.high[6]))
        ## p = 0.005
        self.left_finger_joint.set_position( p, force=5 )
        self.right_finger_joint.set_position( p, force=5 )
        ## if p<0.01:
        ##     self.left_finger_joint.set_torque( -15 )
        ##     self.right_finger_joint.set_torque( -15 )
        ## else:
        ##     self.left_finger_joint.set_torque( 15 )
        ##     self.right_finger_joint.set_torque( 15 )
            

        #temp
        ## self.x_slider.set_velocity(0)
        ## self.y_slider.set_velocity(0)
        ## self.theta_joint.set_velocity(0)
        ## self.left_finger_joint.set_velocity( 0 )
        ## self.right_finger_joint.set_velocity( 0 )

    def calc_state(self):
        self.x, x_dot = self.x_slider.current_position()
        self.y, y_dot = self.y_slider.current_position()
        self.theta, theta_dot = self.theta_joint.current_position()
        self.l_finger, l_finger_dot = self.left_finger_joint.current_position()
        assert( np.isfinite(self.x) )
        assert( np.isfinite(self.y) )
        assert( np.isfinite(self.theta) )
        assert( np.isfinite(self.l_finger) )

        if not np.isfinite(self.x) or not np.isfinite(self.y) or not np.isfinite(self.theta) or not np.isfinite(self.l_finger):
            print("x,y,theta are inf")
            self.x = 0; self.y=0; self.theta=0; self.l_finger=0

        if not np.isfinite(x_dot):
            print("x_dot is inf")
            x_dot = 0
        if not np.isfinite(y_dot):
            print("y_dot is inf")
            y_dot = 0
        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0
        if not np.isfinite(l_finger_dot):
            print("l_finger_dot is inf")
            l_finger_dot = 0
       
        self.to_target_vec = self.peg.current_position() - np.array(self.target.pose().xyz())

        return np.array([
            self.x, x_dot, self.y, y_dot, self.theta, theta_dot, self.l_finger, l_finger_dot, self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2] 
        ])


    def calc_potential(self):
        return -100 * (np.linalg.norm(self.to_target_vec[:2]) + abs(self.to_target_vec[-1]))**2

    
