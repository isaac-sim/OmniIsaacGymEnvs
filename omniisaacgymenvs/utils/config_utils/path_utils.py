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


import os

import carb
from hydra.utils import to_absolute_path


def is_valid_local_file(path):
    return os.path.isfile(path)


def is_valid_ov_file(path):
    import omni.client

    result, entry = omni.client.stat(path)
    return result == omni.client.Result.OK


def download_ov_file(source_path, target_path):
    import omni.client

    result = omni.client.copy(source_path, target_path)

    if result == omni.client.Result.OK:
        return True
    return False


def break_ov_path(path):
    import omni.client

    return omni.client.break_url(path)


def retrieve_checkpoint_path(path):
    # check if it's a local path
    if is_valid_local_file(path):
        return to_absolute_path(path)
    # check if it's an OV path
    elif is_valid_ov_file(path):
        ov_path = break_ov_path(path)
        file_name = os.path.basename(ov_path.path)
        target_path = f"checkpoints/{file_name}"
        copy_to_local = download_ov_file(path, target_path)
        return to_absolute_path(target_path)
    else:
        carb.log_error(f"Invalid checkpoint path: {path}. Does the file exist?")
        return None


def get_experience(headless, enable_livestream, enable_viewport, kit_app):
    if kit_app == '':
        if enable_viewport:
            experience = os.path.abspath(os.path.join('../apps', 'omni.isaac.sim.python.gym.camera.kit'))
        else:
            experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.kit'
            if headless and not enable_livestream:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'
    else:
        experience = kit_app
    return experience
