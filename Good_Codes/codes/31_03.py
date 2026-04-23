import scipy.io as sio
data = sio.loadmat('Q1 _3_SensorData_5dots_diag_NoNoise (1).mat')
sensor_data = data['sensor_data']
print(data.keys())
for k, v in data.items():
    if not k.startswith('_'):
        print(k, type(v), getattr(v, 'shape', ''))