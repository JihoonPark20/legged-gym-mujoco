# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base.legged_robot import LeggedRobot
from .go2.go2_flat_config import Go2FlatCfg, Go2FlatCfgPPO
from .g1.g1_config import G1FlatCfg, G1FlatCfgPPO
from .go2.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO
from .g1.g1_rough_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.utils.task_registry import task_registry

for name in ("go2", "go2_flat"):
    task_registry.register(name, LeggedRobot, Go2FlatCfg(), Go2FlatCfgPPO())
for name in ("g1", "g1_29dof"):
    task_registry.register(name, LeggedRobot, G1FlatCfg(), G1FlatCfgPPO())
task_registry.register("go2_rough", LeggedRobot, Go2RoughCfg(), Go2RoughCfgPPO())
task_registry.register("g1_rough", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO())
