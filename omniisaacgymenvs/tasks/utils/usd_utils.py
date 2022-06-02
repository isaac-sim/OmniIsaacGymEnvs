# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics, UsdLux

def set_drive_type(prim_path, drive_type):
    joint_prim = get_prim_at_path(prim_path)

    # set drive type ("angular" or "linear")
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)
    return drive

def set_drive_target_position(drive, target_value):
    if not drive.GetTargetPositionAttr():
        drive.CreateTargetPositionAttr(target_value)
    else:
        drive.GetTargetPositionAttr().Set(target_value)

def set_drive_target_velocity(drive, target_value):
    if not drive.GetTargetVelocityAttr():
        drive.CreateTargetVelocityAttr(target_value)
    else:
        drive.GetTargetVelocityAttr().Set(target_value)

def set_drive_stiffness(drive, stiffness):
    if not drive.GetStiffnessAttr():
        drive.CreateStiffnessAttr(stiffness)
    else:
        drive.GetStiffnessAttr().Set(stiffness)

def set_drive_damping(drive, damping):
    if not drive.GetDampingAttr():
        drive.CreateDampingAttr(damping)
    else:
        drive.GetDampingAttr().Set(damping)

def set_drive_max_force(drive, max_force):
    if not drive.GetMaxForceAttr():
        drive.CreateMaxForceAttr(max_force)
    else:
        drive.GetMaxForceAttr().Set(max_force)

def set_drive(prim_path, drive_type, target_type, target_value, stiffness, damping, max_force) -> None:
    drive = set_drive_type(prim_path, drive_type)

    # set target type ("position" or "velocity")
    if target_type == "position":
        set_drive_target_position(drive, target_value)
    elif target_type == "velocity":
        set_drive_target_velocity(drive, target_value)

    set_drive_stiffness(drive, stiffness)
    set_drive_damping(drive, damping)
    set_drive_max_force(drive, max_force)

def create_distant_light(prim_path="/World/defaultDistantLight", intensity=5000):
    stage = get_current_stage()
    light = UsdLux.DistantLight.Define(stage, prim_path)
    light.GetPrim().GetAttribute("intensity").Set(intensity)
