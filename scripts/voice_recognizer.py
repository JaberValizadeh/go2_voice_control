#!/usr/bin/env python3
"""Recognize a small offline command vocabulary with Vosk."""

import audioop
import json
import os
import subprocess

import rospy
from std_msgs.msg import String

from go2_voice_control import interpreter


SAMPLE_RATE = 16000
CHUNK_SIZE = 4000

# The vocabulary lives in the interpreter so the phrases we listen for and the
# phrases we can act on cannot drift apart.
GRAMMAR = list(interpreter.RECOGNIZER_GRAMMAR)


def unmute_microphone():
    """Best-effort unmute for ALSA/PulseAudio systems."""
    commands = (
        ["amixer", "sset", "Capture", "100%", "cap"],
        ["pactl", "set-source-mute", "@DEFAULT_SOURCE@", "0"],
    )
    for command in commands:
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def main():
    try:
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except ImportError:
        rospy.logfatal("Python package 'vosk' is not installed")
        return

    rospy.init_node("voice_recognizer")
    unmute_microphone()
    model_path = os.path.expanduser(rospy.get_param("~model_path", ""))
    device = rospy.get_param("~device", "default")
    gain = float(rospy.get_param("~gain", 1.0))
    grammar = rospy.get_param("~grammar", GRAMMAR)
    if isinstance(grammar, str):
        grammar = [item.strip() for item in grammar.split(",") if item.strip()]
    if "[unk]" not in grammar:
        grammar.append("[unk]")

    if not model_path or not os.path.isdir(model_path):
        rospy.logfatal("Set ~model_path to an unpacked Vosk model directory; got '%s'",
                       model_path)
        return

    SetLogLevel(-1)
    rospy.loginfo("Loading Vosk model from %s", model_path)
    recognizer = KaldiRecognizer(Model(model_path), SAMPLE_RATE,
                                 json.dumps(grammar))
    publisher = rospy.Publisher("voice_commands", String, queue_size=10)
    command = ["arecord", "-D", device, "-q", "-f", "S16_LE", "-r",
               str(SAMPLE_RATE), "-c", "1", "-t", "raw"]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
    except OSError as error:
        rospy.logfatal("Could not start arecord: %s", error)
        return

    rospy.loginfo("Listening for Go2 voice commands")
    try:
        while not rospy.is_shutdown():
            data = process.stdout.read(CHUNK_SIZE)
            if not data:
                rospy.logerr("Microphone stream ended unexpectedly")
                break
            if gain != 1.0:
                data = audioop.mul(data, 2, gain)
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "")
                text = text.replace("[unk]", "").strip()
                if text:
                    rospy.loginfo("Heard: '%s'", text)
                    publisher.publish(text)
    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
