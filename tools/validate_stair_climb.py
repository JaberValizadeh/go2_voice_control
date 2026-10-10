#!/usr/bin/env python3
"""Run an isolated, headless ascent of the default solid staircase.

Source the workspace's devel/setup.bash first. Exit zero only after arrival
on the platform and five seconds of stable standing. No microphone is needed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from collections import deque


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/stair_validation'))
    parser.add_argument('--ros-port', type=int, default=11325)
    parser.add_argument('--gazebo-port', type=int, default=11355)
    parser.add_argument('--sim-arg', action='append', default=[],
                        help='extra sim.launch argument, e.g. world:=<file>')
    args = parser.parse_args()
    for port in (args.ros_port, args.gazebo_port):
        with socket.socket() as connection:
            connection.settimeout(0.2)
            if connection.connect_ex(('127.0.0.1', port)) == 0:
                parser.error('port %d is in use; choose an unused test port' % port)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.update(ROS_HOME=str(root / 'ros'), ROS_LOG_DIR=str(root / 'ros/log'),
                      ROS_MASTER_URI='http://localhost:%d' % args.ros_port,
                      GAZEBO_MASTER_URI='http://localhost:%d' % args.gazebo_port)
    import rospy
    from gazebo_msgs.msg import ModelStates, ContactsState
    from sensor_msgs.msg import JointState
    from rosgraph_msgs.msg import Log
    from std_msgs.msg import String

    state = {'pose': None, 'result': None, 'arrived': None}
    contacts, joints = {}, {}
    history = deque(maxlen=200)
    # Full-rate trunk record for the whole climb. `history` keeps joints and
    # contacts but only the last few seconds; comparing one tuning against
    # another needs the entire run, and six floats a tick is cheap enough to
    # keep all of it.
    metrics = []

    def contact(message, leg):
        contacts[leg] = [{'surface': item.collision2_name,
                          'positions': [[v.x, v.y, v.z] for v in item.contact_positions],
                          'normals': [[v.x, v.y, v.z] for v in item.contact_normals]}
                         for item in message.states[:1]]

    def joint(message):
        joints.update(zip(message.name, message.position))

    def model(message):
        if 'unitree_go2' in message.name:
            state['pose'] = message.pose[message.name.index('unitree_go2')]

    def event(message):
        if message.name not in ('/go2_voice_teleop', '/go2_gait'):
            return
        print(message.msg, flush=True)
        if 'Stair run finished: at the top' in message.msg:
            state['arrived'] = rospy.get_time()
        elif ('Stair run finished:' in message.msg or
              'Stair gait blocked:' in message.msg or
              'Robot is down - standing it back up' in message.msg):
            state['result'] = message.msg

    log_path = root / 'gazebo.log'
    publisher = None
    success = False
    with log_path.open('w') as log:
        simulator = subprocess.Popen([
            'roslaunch', 'go2_voice_control', 'sim.launch', 'gui:=false',
            'start_recognizer:=false'] + args.sim_arg, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            while 'gait ready' not in log_path.read_text():
                if simulator.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('simulator failed to start; see ' + str(log_path))
                time.sleep(0.2)
            rospy.init_node('stair_validation', anonymous=True)
            rospy.Subscriber('/gazebo/model_states', ModelStates, model)
            rospy.Subscriber('/rosout', Log, event)
            rospy.Subscriber('/go2_gazebo/joint_states', JointState, joint)
            for leg in ('FL', 'FR', 'RL', 'RR'):
                rospy.Subscriber('/foot_contacts/' + leg, ContactsState, contact, leg)
            publisher = rospy.Publisher('/voice_commands', String, queue_size=1)
            deadline = time.monotonic() + 10
            while state['pose'] is None or not publisher.get_num_connections():
                if time.monotonic() > deadline:
                    raise RuntimeError('robot pose or voice command subscriber missing')
                time.sleep(0.05)
            publisher.publish(String('go up'))
            start = rospy.get_time()
            last_report = -5
            deadline = time.monotonic() + 600
            while state['result'] is None and time.monotonic() < deadline:
                pose = state['pose']
                q = pose.orientation
                roll = math.atan2(2 * (q.w*q.x + q.y*q.z), 1 - 2 * (q.x*q.x + q.y*q.y))
                pitch = math.asin(max(-1, min(1, 2 * (q.w*q.y - q.z*q.x))))
                now = rospy.get_time()
                history.append({'time': now - start,
                                'body': [pose.position.x, pose.position.y, pose.position.z],
                                'orientation': [q.x, q.y, q.z, q.w],
                                'joints': dict(joints), 'contacts': dict(contacts)})
                metrics.append((now - start, pose.position.x, pose.position.y,
                                pose.position.z, math.degrees(roll),
                                math.degrees(pitch)))
                if max(abs(roll), abs(pitch)) > math.radians(35):
                    state['result'] = 'FAIL: robot lost balance'
                if now - start - last_report >= 2:
                    print('t=%.1f xyz=(%.3f, %.3f, %.3f) roll=%.1f pitch=%.1f' % (
                        now-start, pose.position.x, pose.position.y, pose.position.z,
                        math.degrees(roll), math.degrees(pitch)), flush=True)
                    last_report = now - start
                if state['arrived'] is not None and now - state['arrived'] >= 5:
                    success = (pose.position.x >= 4.15 and abs(pose.position.y) < 0.5
                               and pose.position.z >= 0.69
                               and max(abs(roll), abs(pitch)) < math.radians(10))
                    state['result'] = ('PASS: climbed and stood on the platform' if success
                                       else 'FAIL: did not hold the platform stance')
                if simulator.poll() is not None:
                    state['result'] = 'FAIL: simulator exited'
                time.sleep(0.05)
            result = state['result'] or 'FAIL: wall-clock timeout'
            print(result, flush=True)
            pose = state['pose']
            (root / 'result.json').write_text(json.dumps({
                'success': success, 'result': result,
                'simulation_seconds': rospy.get_time() - start,
                'position': [pose.position.x, pose.position.y, pose.position.z]}, indent=2) + '\n')
            (root / 'trace.json').write_text(json.dumps(list(history)) + '\n')
            (root / 'metrics.csv').write_text(
                'time,x,y,z,roll_deg,pitch_deg\n' + ''.join(
                    '%.3f,%.4f,%.4f,%.4f,%.3f,%.3f\n' % row for row in metrics))
        finally:
            if publisher is not None:
                publisher.publish(String('stop'))
            if simulator.poll() is None:
                os.killpg(simulator.pid, signal.SIGINT)
                try:
                    simulator.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(simulator.pid, signal.SIGKILL)
                    simulator.wait()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
