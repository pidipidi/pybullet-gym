from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot
import numpy as np
import os
import pybullet_data, pybullet
import gym, gym.spaces, gym.utils
import time

class Gripper2D(URDFBasedRobot):

    def __init__(self):
        URDFBasedRobot.__init__(self, '../robots/gripper/gripper_description/simple_gripper.urdf', 'cart', action_dim=4, obs_dim=4, basePosition=[0,0,0.025], fixed_base=True, self_collision=False)
        
    def robot_specific_reset(self, bullet_client):
        self._p     = bullet_client
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        #self._p.setPhysicsEngineParameter(solverResidualThreshold=0)
        #self._p.setTimeStep(1./240.)
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

        ## high = [self.jdict['x_slider'].upperLimit,
        ##         self.jdict['y_slider'].upperLimit,
        ##         self.jdict['z_axis_joint'].upperLimit,
        ##         self.jdict['left_finger_joint'].upperLimit]
        ## low = [self.jdict['x_slider'].lowerLimit,
        ##        self.jdict['y_slider'].lowerLimit,
        ##        self.jdict['z_axis_joint'].lowerLimit,
        ##        self.jdict['left_finger_joint'].lowerLimit]
        ## self.observation_space.high[0] = high[0]
        ## self.observation_space.high[2] = high[1]
        ## self.observation_space.high[4] = high[2]
        ## self.observation_space.high[6] = high[3]
        ## self.observation_space.low[0] = low[0]
        ## self.observation_space.low[2] = low[1]
        ## self.observation_space.low[4] = low[2]
        ## self.observation_space.low[6] = low[3]


        p = 0.04
        self.x_slider.reset_current_position(1., 0)
        self.left_finger_joint.reset_current_position( p , 0)
        self.right_finger_joint.reset_current_position( p , 0)
        self.scene.global_step(False)


        # set pos wrt to the peg
        ## u = self.np_random.uniform(low=self.observation_space.low[0]*0.8,
        ##                            high=self.observation_space.high[0]*0.8)
        pos = self.peg.pose().xyz()
        rpy = self.peg.pose().rpy()
        self.theta_joint.reset_current_position( rpy[2] , 0)
        self.x_slider.reset_current_position( pos[0]+0.041*np.cos(rpy[2]) , 0)
        self.y_slider.reset_current_position( pos[1]+0.041*np.sin(rpy[2]) , 0)
        ## self.left_finger_joint.set_torque( -15 )
        ## self.right_finger_joint.set_torque( -15 )
        
        self.x_slider.set_velocity(0)
        self.y_slider.set_velocity(0)
        self.theta_joint.set_velocity(0)
        self.left_finger_joint.set_velocity(0)
        self.right_finger_joint.set_velocity(0)
        self.scene.global_step(False)
        
        ## self.left_finger_joint.set_position( p, force=0.5, maxVelocity=0.0051 )
        ## self.right_finger_joint.set_position( p, force=0.5, maxVelocity=0.0051 )
        #print( "aaaaaaaaaaaaa")
        ## self.scene.global_step(False)
        p = 0.01 #25
        self.left_finger_joint.set_position( p, positionGain=0.5, velocityGain=10., force=3, maxVelocity=0.1 )
        self.right_finger_joint.set_position( p, positionGain=0.5, velocityGain=10., force=3, maxVelocity=0.1 )
        self.scene.global_step(False)
        #print (self._p.getPhysicsEngineParameters())
        ## print( "==============================")
        ## self.scene.global_step(False)
        #from IPython import embed; embed(); sys.exit()


    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0; a[1] = 0; a[2] = 0; a[3] = 0


        p = np.clip(a[0], self.action_space.low[0], self.action_space.high[0])
        p = float(np.clip(p+self.x,
                          self.jdict['x_slider'].lowerLimit,
                          self.jdict['x_slider'].upperLimit))
        self.x_slider.set_position( p , positionGain=0.1, force=2.)

        p = np.clip(a[1], self.action_space.low[1], self.action_space.high[1])
        p = float(np.clip(p+self.y,
                          self.jdict['y_slider'].lowerLimit,
                          self.jdict['y_slider'].upperLimit))
        self.y_slider.set_position( p , positionGain=0.1, force=1.)

        p = np.clip(a[2], self.action_space.low[2], self.action_space.high[2])
        p = float(np.clip(p+self.theta,
                          self.jdict['z_axis_joint'].lowerLimit,
                          self.jdict['z_axis_joint'].upperLimit))
        self.theta_joint.set_position( p , positionGain=0.1, force=2.)

        p = float(np.clip(a[3],
                          self.jdict['left_finger_joint'].lowerLimit,
                          self.jdict['left_finger_joint'].upperLimit))
        ## p = 0.0245
        ## print (p)
        self.left_finger_joint.set_position( p, positionGain=0.5, velocityGain=10., force=3, maxVelocity=0.1 )
        self.right_finger_joint.set_position( p, positionGain=0.5, velocityGain=10., force=3, maxVelocity=0.1 )
        ## if p<0.01:
        ##     self.left_finger_joint.set_torque( -500 )
        ##     self.right_finger_joint.set_torque( -500 )
        ## else:
        ##     self.left_finger_joint.set_position( p, force=200., maxVelocity=1.5 )
        ##     self.right_finger_joint.set_position( p, force=200., maxVelocity=1.5 )
        ##     #self.left_finger_joint.set_torque( 15 )
        ##     #self.right_finger_joint.set_torque( 15 )
            

        #temp
        ## self.x_slider.set_velocity(0)
        ## self.y_slider.set_velocity(0)
        ## self.theta_joint.set_velocity(0)
        ## self.left_finger_joint.set_velocity( 0 )
        ## self.right_finger_joint.set_velocity( 0 )

    def calc_state(self):
        self.x, self.x_dot = self.x_slider.current_position()
        self.y, self.y_dot = self.y_slider.current_position()
        self.theta, self.theta_dot = self.theta_joint.current_position()
        self.l_finger, self.l_finger_dot = self.left_finger_joint.current_position()
        assert( np.isfinite(self.x) )
        assert( np.isfinite(self.y) )
        assert( np.isfinite(self.theta) )
        assert( np.isfinite(self.l_finger) )

        if not np.isfinite(self.x) or not np.isfinite(self.y) or not np.isfinite(self.theta) or not np.isfinite(self.l_finger):
            print("x,y,theta are inf")
            self.x = 0; self.y=0; self.theta=0; self.l_finger=0

        if not np.isfinite(self.x_dot):
            print("x_dot is inf")
            self.x_dot = 0
        if not np.isfinite(self.y_dot):
            print("y_dot is inf")
            self.y_dot = 0
        if not np.isfinite(self.theta_dot):
            print("theta_dot is inf")
            self.theta_dot = 0
        if not np.isfinite(self.l_finger_dot):
            print("l_finger_dot is inf")
            self.l_finger_dot = 0

        pos = self.peg.current_position()[:2] - np.array(self.target.pose().xyz())[:2]
        ## self.to_target_vec[:2] /= 0.25
        ang = pybullet.getEulerFromQuaternion(self.peg.current_orientation())[2] - self.target.pose().rpy()[2]
        self.to_target_vec = np.array([pos[0], pos[1], ang/np.pi])
        ## print (self.peg.current_position()[:2] - np.array(self.target.pose().xyz())[:2], self.to_target_vec)            
        ## print (pybullet.getEulerFromQuaternion(self.peg.current_orientation()), self.peg.current_orientation())
        ## self.to_target_vec[2] = min([ang, ang+np.pi, ang-np.pi])

        return np.array([
            self.l_finger, self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2] 
        ])
        ## return np.array([
        ##     self.x, self.x_dot, self.y, self.y_dot, self.theta, self.theta_dot, self.l_finger, self.l_finger_dot,
        ##     self.to_target_vec[0], self.to_target_vec[1], self.to_target_vec[2] 
        ## ])


    def calc_potential(self):
        return -100. * ( np.sum(self.to_target_vec[:2]**2) + 0.01*abs(self.to_target_vec[-1]) )

    
