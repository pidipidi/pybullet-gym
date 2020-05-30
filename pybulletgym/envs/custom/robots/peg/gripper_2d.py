from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot
import numpy as np
import os
import pybullet_data, pybullet
import gym, gym.spaces, gym.utils
import time
from collections import deque

class Gripper2D(URDFBasedRobot):

    def __init__(self, add_feature='pose', window_size=5):
        self.add_feature = add_feature
        self.window_size = window_size
        if self.add_feature in 'force':
            self.obs_dim = 5
        elif self.add_feature in 'force_window':
            self.obs_dim = 2*window_size + 3            
        else:
            self.obs_dim = 4

        URDFBasedRobot.__init__(self, '../robots/gripper/gripper_description/simple_gripper.urdf', 'cart', action_dim=4, obs_dim=self.obs_dim, basePosition=[0,0,0.025], fixed_base=True, self_collision=False)

        self.max_obs = np.zeros((self.obs_dim,))
        self.min_obs = np.zeros((self.obs_dim,))
        
        
    def robot_specific_reset(self, bullet_client):
        self._p     = bullet_client
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        #self._p.setPhysicsEngineParameter(solverResidualThreshold=0)
        #self._p.setTimeStep(1./240.)
        ## from IPython import embed; embed(); sys.exit()
        
        # A centering constraint for the gripper 
        c = self._p.createConstraint(self.objects,
                                     4,
                                     self.objects,
                                     6,
                                     jointType=self._p.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)


        for j in range(self._p.getNumJoints(self.objects)):
            self._p.changeDynamics(self.objects, j, linearDamping=0, angularDamping=0)
        
        self.target = self.parts["target"]
        self.peg    = self.parts["peg"]
        self.peg_handle = self.parts["peg_right"]
        
        self.x_slider    = self.jdict["x_slider"]
        self.y_slider    = self.jdict["y_slider"]
        self.theta_joint = self.jdict["z_axis_joint"]
        self.left_finger_joint  = self.jdict["left_finger_joint"]
        self.right_finger_joint = self.jdict["right_finger_joint"]

        self.action_scale = [self.jdict['x_slider'].jointMaxVelocity,
                             self.jdict['y_slider'].jointMaxVelocity,
                             self.jdict['z_axis_joint'].jointMaxVelocity,
                             self.jdict['left_finger_joint'].jointMaxVelocity]
        ## self.action_space = gym.spaces.Box(-np.array(high), np.array(high))
        
        ## if 'force' in self.add_feature:
        ##     self.theta_joint.enable_ft_sensor()
            ## self.left_finger_joint.enable_ft_sensor()
            ## self.right_finger_joint.enable_ft_sensor()

        p = 0.02
        self.x_slider.reset_current_position(1., 0)
        self.left_finger_joint.reset_current_position( p , 0)
        self.right_finger_joint.reset_current_position( p , 0)
        self.scene.global_step(False)


        # set pos wrt to the peg
        pos = self.peg.pose().xyz()
        rpy = self.peg.pose().rpy()
        self.theta_joint.reset_current_position( rpy[2] , 0)
        self.x_slider.reset_current_position( pos[0]+0.041*np.cos(rpy[2]) , 0)
        self.y_slider.reset_current_position( pos[1]+0.041*np.sin(rpy[2]) , 0)
        ## self.left_finger_joint.set_torque( -15 )
        ## self.right_finger_joint.set_torque( -15 )        
        self.scene.global_step(False)
        
        p = 0.01 #25
        self.left_finger_joint.set_position( p, positionGain=0.001, velocityGain=0.01, force=50., maxVelocity=0.1 )
        self.right_finger_joint.set_position( p, positionGain=0.001, velocityGain=0.01, force=50., maxVelocity=0.1 )
        self.scene.global_step(False)
        self.scene.global_step(False)
        self.scene.global_step(False)
        self.scene.global_step(False)
        self.scene.global_step(False)
        self.scene.global_step(False)
        
        self._p.setGravity(0, -9.8, 0)
        self.scene.global_step(False)

        # sensor queue
        if self.add_feature in 'force':
            _, _, _, self.theta_t = self.theta_joint.get_full_state()
            self.queue = deque([self.theta_t/0.8]*self.window_size, self.window_size)            
        elif self.add_feature in 'force_window':
            self.l_finger, _ = self.left_finger_joint.current_position()
            _, _, _, self.theta_t = self.theta_joint.get_full_state()
            self.queue = deque([[self.l_finger/0.04, self.theta_t/0.8]]*self.window_size, self.window_size)
        
        obs = self.calc_state()

        self.des_pos = [pos[0]+0.041*np.cos(rpy[2]),
                        pos[1]+0.041*np.sin(rpy[2]),
                        rpy[2], p, p]
        self.des_vel = [0,0,0,0,0]



    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0; a[1] = 0; a[2] = 0; a[3] = 0

        #from IPython import embed; embed(); sys.exit()
        grav_comp_torque = self._p.calculateInverseDynamics(self.objects,
                                                            [self.x, self.y, self.theta,
                                                             self.l_finger, self.r_finger],
                                                            [0] * 5,
                                                            [0,9.8,0,0,0])
        ## a *= 0
        ## a[2] = -.4
        self.des_vel = np.array(a) * self.action_scale 

        self.des_pos[0] = float(np.clip(self.des_vel[0]+self.des_pos[0],
                          self.jdict['x_slider'].lowerLimit,
                          self.jdict['x_slider'].upperLimit))

        self.des_pos[1] = float(np.clip(self.des_vel[1]+self.des_pos[1],
                          self.jdict['y_slider'].lowerLimit,
                          self.jdict['y_slider'].upperLimit))

        self.des_pos[2] = float(np.clip(self.des_vel[2]+self.des_pos[2],
                          self.jdict['z_axis_joint'].lowerLimit,
                          self.jdict['z_axis_joint'].upperLimit))
        #a[3] = 0
        self.des_pos[3] = float(np.clip(self.des_vel[3]+self.des_pos[3],
                          self.jdict['left_finger_joint'].lowerLimit,
                          self.jdict['left_finger_joint'].upperLimit))

        # for 0.01kg peg mass
        grav_comp_scale = 1.
        # for 0.01kg peg mass
        #grav_comp_scale = 80.155

        self.x_slider.set_torque( grav_comp_torque[0] * grav_comp_scale )
        self.y_slider.set_torque( grav_comp_torque[1] * grav_comp_scale )
        self.theta_joint.set_torque( grav_comp_torque[2] * grav_comp_scale )
        self.left_finger_joint.set_torque( grav_comp_torque[3] * grav_comp_scale )
        self.right_finger_joint.set_torque( grav_comp_torque[4] * grav_comp_scale )
        
        self.x_slider.set_position( self.des_pos[0] , positionGain=0.2, velocityGain=10., force=30.)
        self.y_slider.set_position( self.des_pos[1] , positionGain=0.2, velocityGain=10., force=40.)
        self.theta_joint.set_position( self.des_pos[2] , positionGain=0.1, force=0.8, maxVelocity=3.14)         
        self.left_finger_joint.set_position( self.des_pos[3], positionGain=0.001, velocityGain=0.01, force=50, maxVelocity=0.1 )
        self.right_finger_joint.set_position( self.des_pos[3], positionGain=0.001, velocityGain=0.01, force=50, maxVelocity=0.1 )
        ## print (self.l_finger, self.r_finger, self.l_finger-self.r_finger)



    def calc_state(self):
        self.x, self.x_dot = self.x_slider.current_position()
        self.y, self.y_dot = self.y_slider.current_position()
        self.theta, self.theta_dot, f, self.theta_t = self.theta_joint.get_full_state()
        self.l_finger, self.l_finger_dot, _, self.l_finger_t = self.left_finger_joint.get_full_state()
        self.r_finger, self.r_finger_dot, _, self.r_finger_t = self.right_finger_joint.get_full_state()
        assert( np.isfinite(self.x) )
        assert( np.isfinite(self.y) )
        assert( np.isfinite(self.theta) )
        assert( np.isfinite(self.l_finger) )

        ## print (self.theta_joint.get_joint_info())
        ## sys.exit()
        ## print ("{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}".format(f[0], f[1], f[2], f[3], f[4], f[5], self.theta_t))
        #print ("{:.4}, {:.4}, {:.4}, {:.4}".format(f[3], f[4], f[5], self.theta_t))
        ## print ("{:.4}, {:.4}, {:.4}".format(f[3], f[4], f[5]))

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
        ang = pybullet.getEulerFromQuaternion(self.peg.current_orientation())[2] - self.target.pose().rpy()[2]
        self.to_target_vec = np.array([pos[0], pos[1], ang/np.pi])

        if self.add_feature in 'force':
            # applied joint motor torque
            self.queue.append(self.theta_t/0.8)
            obs = np.array([
                self.l_finger/0.04,
                np.mean(self.queue),
                self.to_target_vec[0]/0.05,
                self.to_target_vec[1]/0.05,
                self.to_target_vec[2]/3.14 
                ])
            
        elif self.add_feature in 'force_window':

            self.queue.append([self.l_finger/0.04,
                               self.theta_t/0.8])
            obs = np.array(np.array(self.queue).T.flatten().tolist()
                           + [self.to_target_vec[0]/0.05,
                              self.to_target_vec[1]/0.05,
                              self.to_target_vec[2]/3.14 ])
        else:
            obs = np.array([ self.l_finger/0.04,
                             self.to_target_vec[0]/0.05,
                             self.to_target_vec[1]/0.05,
                             self.to_target_vec[2]/3.14 ])

        ## self.max_obs = np.maximum( self.max_obs, obs)
        ## self.min_obs = np.minimum( self.min_obs, obs)
        ## print( self.max_obs[1], self.min_obs[1], self.theta_t)
        return obs

    def calc_potential(self):        
        return -100. * ( np.linalg.norm(self.to_target_vec[:2]) + 0.01*abs(self.to_target_vec[-1]) )

    def release_reward(self):
        if abs(self.to_target_vec[0]) < 0.005 and \
          abs(self.to_target_vec[1]) < 0.005 and \
          abs(self.to_target_vec[-1]) < np.pi*5./180:
            return 1.
        elif (abs(self.to_target_vec[0]) > 0.02 or \
          abs(self.to_target_vec[1]) > 0.02) and self.l_finger > 0.02:
            return -0.1
          ##   and\
          ## self.l_finger + self.r_finger > 0.03 :
          ##   return 1. #*self.l_finger
          ## elif self.l_finger + self.r_finger > 0.02:
          ##     return -1.
            ## return -1.*(self.l_finger + self.r_finger)**2
        else:
            return 0.
        
    
