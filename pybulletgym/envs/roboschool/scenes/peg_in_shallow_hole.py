import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

from .scene_bases import Scene
import pybullet
import pybullet_data
import numpy as np


class PegInShallowHoleScene(Scene):
	multiplayer = False
	zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
	## stadium_halflen   = 105*0.25	# FOOBALL_FIELD_HALFLEN
	## stadium_halfwidth = 50*0.25	 # FOOBALL_FIELD_HALFWID
	stadiumLoaded = 0
	stadium = None

	def episode_restart(self, bullet_client, robot=None):
		self._p = bullet_client
		Scene.episode_restart(self, bullet_client)
		if self.stadiumLoaded == 0:
			self.stadiumLoaded = 1

			self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
			self.plane = self._p.loadURDF("plane.urdf")
			full_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", "peg", "peg_description", "shallow_hole.urdf")
			self.stadium = self._p.loadURDF(full_path, basePosition=[0,0,0.025], useFixedBase=True)#, flags=pybullet.URDF_USE_SELF_COLLISION)
			full_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", "peg", "peg_description", "cuboid_thin_peg.urdf")
			self.peg = self._p.loadURDF(full_path, basePosition=[0,0,0.025],
                                        useFixedBase=True,
                                        flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

        # remove gripper once
		if robot is not None and 'x_slider' in robot.jdict.keys():
			robot.jdict['x_slider'].reset_current_position(1e+10 , 0)
			robot.jdict['left_finger_joint'].reset_current_position( 0.03 , 0)
			robot.jdict['right_finger_joint'].reset_current_position( 0.03 , 0)
			self.global_step()

		position = [ self.np_random.uniform(low=-0.08,
		                                        high=0.08),
                    self.np_random.uniform(low=-0.08,
                                           high=0.08),
                    0.025 ]
		orientation = [0, 0, self.np_random.uniform(low=-np.pi,
                                                        high=np.pi)]
		## orientation = self._p.getQuaternionFromEuler(orientation)
		## self._p.resetBasePositionAndOrientation(self.peg, position,
        ##                                         orientation)

		if robot is not None:                   	
            #;from IPython import embed; embed(); sys.exit()
			robot.jdict['peg_x_slider'].reset_position(position[0],0)
			robot.jdict['peg_y_slider'].reset_position(position[1],0)
			robot.jdict['peg_z_axis_joint'].reset_position(orientation[-1],0)
			## print(self._p.getJointState(self.peg, 0))
            #self._p.setJointMotorControlArray(self.peg_joint, )
			
		self.global_step()

		## print(self._p.getBaseVelocity(self.peg))
		## from IPython import embed; embed(); sys.exit()
