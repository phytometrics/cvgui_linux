import subprocess
import re



# https://github.com/azeam/camset/blob/master/camset/v4l2control.py
# ここらへんあありが参考になるかも。
def get_devices():
    """
    :return: dict
    """

    output = subprocess.run(["/usr/bin/v4l2-ctl --list-devices"], shell=True, capture_output=True)
    assert output.returncode == 0, "v4l2-ctl command possibly failed check whether v4l-utility is installed"

    camera_ids = output.stdout.decode("utf-8")
    camera_ids = camera_ids.splitlines()  # to list
    d = {}
    camerano = 0
    for s in camera_ids:
        if len(s) == 0:  # empty line for next camera
            camerano += 1
            continue
        if s[-1] == ":":  # camera info
            d["camera_" + str(camerano)] = {}
            d["camera_"+str(camerano)]["camera_name"] = s[:-1]
            d["camera_" + str(camerano)]["io_name"] = []
        if s.startswith("\t"):  # /dev/videoN
            # just use the first /dev/videoN, may remove for future
            if len(d["camera_" + str(camerano)]["io_name"]) == 0:
                d["camera_" + str(camerano)]["io_name"].append(s[1:])
                d["camera_" + str(camerano)]["io_name"] = d["camera_" + str(camerano)]["io_name"][0]

    # remove surface camera
    d = {key: val for key, val in d.items() if val["camera_name"].find("PCI:0000") == -1}
    return d

def get_resolution(camera_dict):
    """
    currently hardcoded for MJPG and YUYV only
    """
    for key, val in camera_dict.items():
        s = "/usr/bin/v4l2-ctl --device {} --list-formats-ext".format(camera_dict[key]["io_name"])
        output = subprocess.run([s], shell=True, capture_output=True)
        assert output.returncode == 0, "{} failed".format(s)

        camera_dict[key]["formats"] = {}  # fourcc, width, height, fps

        camera_info = output.stdout.decode("utf-8")
        camera_info = camera_info.splitlines()  # to list

        i = 0
        fourcc = ""
        width = ""
        height = ""
        fps = ""

        for s in camera_info:

            if s.find("MJPG") != -1:
                fourcc = "MJPG"
            elif s.find("YUYV") != -1:
                fourcc = "YUYV"
            else:
                s = s.replace("\t", "")

                if s.startswith("Size"):  # size: dicrete 1920x1080

                    wh = s.split(" ")[-1].split("x")

                    width = wh[0]
                    height = wh[1]

                elif s.startswith("Interval"):  # interval: discrete 0.0033s (30.000 fps)

                    if "config_{}".format(i) not in camera_dict[key]["formats"]:
                        camera_dict[key]["formats"]["config_{}".format(str(i).zfill(2))] = {}

                    fps = int(s.split(" ")[-2].split(".")[0][1:])  #  (30.000

                    camera_dict[key]["formats"]["config_{}".format(str(i).zfill(2))]["fourcc"] = fourcc
                    camera_dict[key]["formats"]["config_{}".format(str(i).zfill(2))]["height"] = height
                    camera_dict[key]["formats"]["config_{}".format(str(i).zfill(2))]["width"] = width
                    camera_dict[key]["formats"]["config_{}".format(str(i).zfill(2))]["fps"] = fps

                    i += 1
    return camera_dict











