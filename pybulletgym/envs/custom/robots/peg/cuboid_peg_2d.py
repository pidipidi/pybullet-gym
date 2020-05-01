from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot
import numpy as np
import os
import pybullet_data

class CuboidPeg2D(URDFBasedRobot):

    def __init__(self):
        URDFBasedRobot.__init__(self, '../urdf/cuboid_peg.urdf', 'cart', action_dim=3, obs_dim=9, basePosition=[0,0,0.0235], fixed_base=True)

    def robot_specific_reset(self, bullet_client):
        self._p     = bullet_client
        self.target = self.parts["target"]
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        #self._p.setGravity(0, 0, -10)

        ## from IPython import embed; embed(); sys.exit()
        self.plane = self._p.loadURDF("plane.urdf")
        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "urdf", "hole.urdf")
        self.arena = self._p.loadURDF(full_path, basePosition=[0,0,0.0235], useFixedBase=True)
        
        self.x_slider    = self.jdict["x_slider"]
        self.y_slider    = self.jdict["y_slider"]
        self.theta_joint = self.jdict["z_axis_joint"]
        u = self.np_random.uniform(low=-.45, high=.45)
        self.x_slider.reset_current_position( u , 0)
        u = self.np_random.uniform(low=-.45, high=.45)
        self.y_slider.reset_current_position( u , 0)
        u = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.theta_joint.reset_current_position( u , 0)
        
        self.x_slider.set_velocity(0)
        self.y_slider.set_velocity(0)
        self.theta_joint.set_velocity(0)
        ## print (self.jdict.keys())
        ## print (self.jdict)
        ## from IPython import embed; embed(); sys.exit()
        

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0; a[1] = 0; a[2] = 0
            
        self.x_slider.set_velocity( float(np.clip(a[0], -self.jdict['x_slider'].jointMaxVelocity, self.jdict['x_slider'].jointMaxVelocity)) )
        self.y_slider.set_velocity( float(np.clip(a[1], -self.jdict['y_slider'].jointMaxVelocity, self.jdict['y_slider'].jointMaxVelocity)) )
        self.theta_joint.set_velocity( float(np.clip(a[2], -self.jdict['z_axis_joint'].jointMaxVelocity, self.jdict['z_axis_joint'].jointMaxVelocity)) )

    def calc_state(self):
        self.x, x_dot = self.x_slider.current_position()
        self.y, y_dot = self.y_slider.current_position()
        self.theta, theta_dot = self.theta_joint.current_position()
        assert( np.isfinite(self.x) )
        assert( np.isfinite(self.y) )
        assert( np.isfinite(self.theta) )

        if not np.isfinite(self.x) or not np.isfinite(self.y) or not np.isfinite(self.theta):
            print("x,y,theta are inf")
            self.x = 0; self.y=0; self.theta=0

        if not np.isfinite(x_dot):
            print("x_dot is inf")
            x_dot = 0
        if not np.isfinite(y_dot):
            print("y_dot is inf")
            y_dot = 0
        if not np.isfinite(theta_dot):
            print("x_dot is inf")
            theta_dot = 0

        self.to_target_vec = np.array([self.x, self.y]) - np.array(self.target.pose().xyz()[:2])
        self.to_target_theta = self.theta - self.target.pose().rpy()[2]

        return np.array([
            self.x, x_dot, self.y, y_dot, self.theta, theta_dot, self.to_target_vec[0], self.to_target_vec[1], self.to_target_theta
        ])


    def calc_potential(self):
        return -100 * ((np.linalg.norm(self.to_target_vec)) + (abs(self.to_target_theta)))**2
