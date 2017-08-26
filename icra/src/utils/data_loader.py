import scipy.io as sio
import time
import copy

class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

        # object.__setattr__(self, sphero_data, )

    # Freeze fields so new ones cannot be set.
    # def __setattr__(self, key, value):
    #     if not hasattr(self, key):
    #         raise AttributeError("%r has no attribute %s" % (self, key))
    #     object.__setattr__(self, key, value)

# template for experimental data
class SpheroLoader(BundleType):
    def __init__(self):

        self.sphero_data = sio.loadmat("../data/sphero.mat")

        experiment_params = {
                'T': None,
                'pos': {
                    'x': None,
                    'y': None,
                },
                'vel': {
                    'x': None,
                    'y': None,
                },
                'vel_est': {
                    'x': None,
                    'y': None,
                }
            }

        self.experiment = {
            'robot_I': experiment_params,
            'robot_II': experiment_params,
            'robot_III': experiment_params,
        }

        BundleType.__init__(self, self.load_expt_I())
        BundleType.__init__(self, self.load_expt_II())
        BundleType.__init__(self, self.load_expt_III())

    def load_expt_I(self):
        experiment_I     = copy.deepcopy(self.experiment)
        """
        populate values for each robot in
        experiment I
        print(sphero_data['exp']['expI'][0][0]['pos_x1'])
        """
        experiment_I['robot_I']['T'] = self.sphero_data['exp']['expI'][0][0]['T1']
        experiment_I['robot_I']['pos']['x'] = self.sphero_data['exp']['expI'][0][0]['pos_x1']
        experiment_I['robot_I']['pos']['y'] = self.sphero_data['exp']['expI'][0][0]['pos_y1']
        experiment_I['robot_I']['vel']['x'] = self.sphero_data['exp']['expI'][0][0]['cmd_vel_x1']
        experiment_I['robot_I']['vel']['y'] = self.sphero_data['exp']['expI'][0][0]['cmd_vel_y1']
        experiment_I['robot_I']['vel_est']['x'] = self.sphero_data['exp']['expI'][0][0]['est_vel_x1']
        experiment_I['robot_I']['vel_est']['y'] = self.sphero_data['exp']['expI'][0][0]['est_vel_y1']

        # experiment II
        experiment_I['robot_II']['T'] = self.sphero_data['exp']['expI'][0][0]['T2']
        experiment_I['robot_II']['pos']['x'] = self.sphero_data['exp']['expI'][0][0]['pos_x2']
        experiment_I['robot_II']['pos']['y'] = self.sphero_data['exp']['expI'][0][0]['pos_y2']
        experiment_I['robot_II']['vel']['x'] = self.sphero_data['exp']['expI'][0][0]['cmd_vel_x2']
        experiment_I['robot_II']['vel']['y'] = self.sphero_data['exp']['expI'][0][0]['cmd_vel_y2']
        experiment_I['robot_II']['vel_est']['x'] = self.sphero_data['exp']['expI'][0][0]['est_vel_x2']
        experiment_I['robot_II']['vel_est']['y'] = self.sphero_data['exp']['expI'][0][0]['est_vel_y2']

        # experiment III
        experiment_I['robot_III']['T'] = self.sphero_data['exp']['expI'][0][0]['T3']
        experiment_I['robot_III']['pos']['x'] = self.sphero_data['exp']['expI'][0][0]['pos_x3']
        experiment_I['robot_III']['pos']['y'] = self.sphero_data['exp']['expI'][0][0]['pos_y3']
        experiment_I['robot_III']['vel']['x'] = self.sphero_data['exp']['expI'][0][0]['cmd_vel_x3']
        experiment_I['robot_III']['vel']['y'] = self.sphero_data['exp']['expI'][0][0]['cmd_vel_y3']
        experiment_I['robot_III']['vel_est']['x'] = self.sphero_data['exp']['expI'][0][0]['est_vel_x3']
        experiment_I['robot_III']['vel_est']['y'] = self.sphero_data['exp']['expI'][0][0]['est_vel_y3']

        return experiment_I


    def load_expt_II(self):
        experiment_II = copy.deepcopy(self.experiment)
        """
            data for expt II
        """
        experiment_II['robot_I']['T'] = self.sphero_data['exp']['expII'][0][0]['T1']
        experiment_II['robot_I']['pos']['x'] = self.sphero_data['exp']['expII'][0][0]['pos_x1']
        experiment_II['robot_I']['pos']['y'] = self.sphero_data['exp']['expII'][0][0]['pos_y1']
        experiment_II['robot_I']['vel']['x'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_x1']
        experiment_II['robot_I']['vel']['y'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_y1']
        experiment_II['robot_I']['vel_est']['x'] = self.sphero_data['exp']['expII'][0][0]['est_vel_x1']
        experiment_II['robot_I']['vel_est']['y'] = self.sphero_data['exp']['expII'][0][0]['est_vel_y1']

        # experiment II
        experiment_II['robot_II']['T'] = self.sphero_data['exp']['expII'][0][0]['T2']
        experiment_II['robot_II']['pos']['x'] = self.sphero_data['exp']['expII'][0][0]['pos_x2']
        experiment_II['robot_II']['pos']['y'] = self.sphero_data['exp']['expII'][0][0]['pos_y2']
        experiment_II['robot_II']['vel']['x'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_x2']
        experiment_II['robot_II']['vel']['y'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_y2']
        experiment_II['robot_II']['vel_est']['x'] = self.sphero_data['exp']['expII'][0][0]['est_vel_x2']
        experiment_II['robot_II']['vel_est']['y'] = self.sphero_data['exp']['expII'][0][0]['est_vel_y2']

        # experiment III
        experiment_II['robot_III']['T'] = self.sphero_data['exp']['expIII'][0][0]['T3']
        experiment_II['robot_III']['pos']['x'] = self.sphero_data['exp']['expIII'][0][0]['pos_x3']
        experiment_II['robot_III']['pos']['y'] = self.sphero_data['exp']['expIII'][0][0]['pos_y3']
        experiment_II['robot_III']['vel']['x'] = self.sphero_data['exp']['expIII'][0][0]['cmd_vel_x3']
        experiment_II['robot_III']['vel']['y'] = self.sphero_data['exp']['expIII'][0][0]['cmd_vel_y3']
        experiment_II['robot_III']['vel_est']['x'] = self.sphero_data['exp']['expIII'][0][0]['est_vel_x3']
        experiment_II['robot_III']['vel_est']['y'] = self.sphero_data['exp']['expIII'][0][0]['est_vel_y3']

        return experiment_II

    def load_expt_III(self):
        experiment_III   = copy.deepcopy(self.experiment)
        """
            data for expt III
        """
        experiment_III['robot_I']['T'] = self.sphero_data['exp']['expII'][0][0]['T1']
        experiment_III['robot_I']['pos']['x'] = self.sphero_data['exp']['expII'][0][0]['pos_x1']
        experiment_III['robot_I']['pos']['y'] = self.sphero_data['exp']['expII'][0][0]['pos_y1']
        experiment_III['robot_I']['vel']['x'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_x1']
        experiment_III['robot_I']['vel']['y'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_y1']
        experiment_III['robot_I']['vel_est']['x'] = self.sphero_data['exp']['expII'][0][0]['est_vel_x1']
        experiment_III['robot_I']['vel_est']['y'] = self.sphero_data['exp']['expII'][0][0]['est_vel_y1']

        # experiment II
        experiment_III['robot_II']['T'] = self.sphero_data['exp']['expII'][0][0]['T2']
        experiment_III['robot_II']['pos']['x'] = self.sphero_data['exp']['expII'][0][0]['pos_x2']
        experiment_III['robot_II']['pos']['y'] = self.sphero_data['exp']['expII'][0][0]['pos_y2']
        experiment_III['robot_II']['vel']['x'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_x2']
        experiment_III['robot_II']['vel']['y'] = self.sphero_data['exp']['expII'][0][0]['cmd_vel_y2']
        experiment_III['robot_II']['vel_est']['x'] = self.sphero_data['exp']['expII'][0][0]['est_vel_x2']
        experiment_III['robot_II']['vel_est']['y'] = self.sphero_data['exp']['expII'][0][0]['est_vel_y2']

        # experiment III
        experiment_III['robot_III']['T'] = self.sphero_data['exp']['expIII'][0][0]['T3']
        experiment_III['robot_III']['pos']['x'] = self.sphero_data['exp']['expIII'][0][0]['pos_x3']
        experiment_III['robot_III']['pos']['y'] = self.sphero_data['exp']['expIII'][0][0]['pos_y3']
        experiment_III['robot_III']['vel']['x'] = self.sphero_data['exp']['expIII'][0][0]['cmd_vel_x3']
        experiment_III['robot_III']['vel']['y'] = self.sphero_data['exp']['expIII'][0][0]['cmd_vel_y3']
        experiment_III['robot_III']['vel_est']['x'] = self.sphero_data['exp']['expIII'][0][0]['est_vel_x3']
        experiment_III['robot_III']['vel_est']['y'] = self.sphero_data['exp']['expIII'][0][0]['est_vel_y3']

        return experiment_III

    # def __call__(self):
    #     exp_I = self.load_expt_I()
    #     exp_I = BundleType.__init__(self, self.load_expt_I())
    #
    #     exp_II = self.load_expt_II()
    #     exp_II = BundleType.__init__(self, exp_II)
    #
    #     exp_III = self.load_expt_III()
    #     exp_III = BundleType.__init__(self, exp_III)
    #
    #     experiments = {'experiment_I': exp_I, 'experiment_II': exp_II, 'experiment_III': exp_III}
    #
    #     print(exp_I)
    #     return experiments

sphero_dicts = SpheroLoader()#.__call__()

# for k, v in sphero_dicts['experiment_I'].items():
#     print(k, v)
# sphero_dicts#.__call__()

print(sphero_dicts.__dict__.keys(), type(sphero_dicts))
print(sphero_dicts.experiment['robot_I'])
