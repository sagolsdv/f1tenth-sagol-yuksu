import subprocess, time, os
while True:
    if subprocess.check_output(('ifconfig', 'eth0')).decode('utf-8').find('192.168.0.100') < 0:
        os.system('ifconfig eth0 192.168.0.100')

    time.sleep(1)
