# NOTE assume that agentA is in the middle of each room
import os
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Param import *
import numpy as np

# TODO: check read order
class EnvDataset(Dataset):
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.env_files = os.listdir(self.env_dir)
        self.env_files = [f for f in self.env_files if not f.startswith(".")]  # remove .DS_Store
        #self.colors = ["red","green", "blue", "purple", "yellow", "grey"]
        #self.shapes = ["key","ball","box"]
        #跟后面的向量映射有关
        #self.directions = ["up", "down", "left", "right"]
        #self.color_mapping = {color: i for i, color in enumerate(self.colors)}
        #self.shape_mapping = {shape: i for i, shape in enumerate(self.shapes)}
        #self.direction_mapping = {direction: i for i, direction in enumerate(self.directions)}
        #self.task = task  # TODO if task == "GuessRoom", then __getitem__ return vectors only relevant to guessing room task

    def __getitem__(self, item):
        with open("{}/obs{}.txt".format(self.env_dir, item), 'rb') as f:
            #env_info = json.load(f)
            #context=f.read()
            #env_info = eval(context)
            data=np.load(file=f)
            return data
            
            #num_room, num_obs, num_gate, obs_info, gate_info =self.guess_room_env_info(env_info)
        #if self.task == "GuessRoom":
        #    return  num_room,obs_info,gate_info
        #elif self.task == "GuessGate":
        #    return num_room, num_gate, gate_info
        #elif self.task == "Navigation":
        #    return num_room, num_gate, obs_info, gate_info, item

    def __len__(self):
        return len(self.env_files) if Param.is_dynamic_data is False else Param.dynamic_datasize
    def PO_info(self, env_info, room_id, agentA_pos, agentA_dir):
        '''
        NOTE assume agentA is in the center of this room
        TODO not sure if obs and gate should be dealt with separately
        TODO is it necessary to add type in vectors ?
        :param env_info:
        :return:
        '''
        
        # for room_str, room_info in env_info.items():  
        room_str = str(room_id); room_info = env_info[room_str]
        gates_info = room_info["gates"]
        obs_info = room_info["obstacles"]
        #agentAdir---0:up 1:right 2:down 3:left
        #3*3
        top=agentA_pos[0];down=agentA_pos[0]
        left=agentA_pos[1];right=agentA_pos[1]
        if agentA_dir in [0,2]:   
            left=max(left-1,0)
            right=min(right+1,5)
            if agentA_dir==0:
                top=max(top-2,0)
            else:
                down=min(down+2,5)
        elif agentA_dir in [1,3]:
            top=max(top-1,0)
            down=min(down+1,5)
            if agentA_dir==1:
                right=min(right+2,5)
            else:
                left=max(left-2,0)
        gate_vectors = []; obs_vectors = []
        for obs_id in range(len(obs_info.keys())):  # NOTE make sure that the obs is ordered
            obs_str = str(obs_id); cur_obs_info = obs_info[obs_str]
            cur_obs_pos=cur_obs_info["pos"]
            if cur_obs_pos[0] in range(top,down+1) and cur_obs_pos[1] in range(left,right+1):
                cur_vector,_= self._get_obs_vector(agentA_pos, cur_obs_info)   
                obs_vectors.append(cur_vector)
        for gate_id in range(len(gates_info.keys())):  # NOTE make sure that the gate is ordered
            gate_str = str(gate_id); cur_gate_info = gates_info[gate_str]
            cur_gate_pos=cur_gate_info["pos"]
            cur_dir_index = self.direction_mapping[cur_gate_info["direction"]]
            if (cur_dir_index==0 and top==0) or (cur_dir_index==1 and down==5) :
                if cur_gate_pos in range(left,right+1):
                    cur_vector,_ = self._get_gate_vector(agentA_pos, cur_gate_info) 
                    gate_vectors.append(cur_vector)
            elif (cur_dir_index==2 and left==0) or (cur_dir_index==3 and right==5):
                if cur_gate_pos in range(top,down+1):
                    cur_vector,_ = self._get_gate_vector(agentA_pos, cur_gate_info) 
                    gate_vectors.append(cur_vector)
        gate_array = np.array(gate_vectors, dtype=float)
        obs_array = np.array(obs_vectors, dtype=float)
        if gate_array.shape[0] == 0: gate_array = np.zeros((Param.max_subgate_num, Param.gate_feat_in_num))
        if obs_array.shape[0] == 0: obs_array = np.zeros((Param.max_subobs_num, Param.obs_feat_in_num))
        #num_obs.append(obs_array.shape[0]); num_gate.append(gate_array.shape[0])
        # ---- PAD ----
        #2*3,4*4
        gate_array = np.pad(gate_array, ((0, Param.max_subgate_num - gate_array.shape[0]), (0, 0)))
        obs_array = np.pad(obs_array, ((0, Param.max_subobs_num - obs_array.shape[0]), (0, 0)))
        return gate_array,obs_array
"""
    def guess_room_env_info(self, env_info):
        '''
        NOTE assume agentA is in the center of this room
        TODO not sure if obs and gate should be dealt with separately
        TODO is it necessary to add type in vectors ?
        :param env_info:
        :return:
        '''
        # res = {}
        res_obs = []; res_gate = []
        num_obs = []; num_gate = []
        # for room_str, room_info in env_info.items():
        for room_id in range(len(env_info.keys())):  # NOTE make sure that the room is ordered
            room_str = str(room_id); room_info = env_info[room_str]
            gates_info = room_info["gates"]
            obs_info = room_info["obstacles"]
            room_pos = room_info["pos"]
            room_pos = [int(i) for i in room_pos]
            agentA_pos = ((room_pos[0] + room_pos[1])//2 - room_pos[0], (room_pos[2] + room_pos[3])//2 - room_pos[2])
            # room_vectors = []
            gate_vectors = []; obs_vectors = []
            # ---- gate ----
            # for gate_str, cur_gate_info in gates_info.items():
            for gate_id in range(len(gates_info.keys())):  # NOTE make sure that the gate is ordered
                gate_str = str(gate_id); cur_gate_info = gates_info[gate_str]
                cur_vector,_ = self._get_gate_vector(agentA_pos, cur_gate_info)
                gate_vectors.append(cur_vector)
            # ---- obs ----
            # for obs_str, cur_obs_info in obs_info.items():
            for obs_id in range(len(obs_info.keys())):  # NOTE make sure that the obs is ordered
                obs_str = str(obs_id); cur_obs_info = obs_info[obs_str]
                cur_vector,_ = self._get_obs_vector(agentA_pos, cur_obs_info)
                obs_vectors.append(cur_vector)
            gate_array = np.array(gate_vectors, dtype=float)
            obs_array = np.array(obs_vectors, dtype=float)
            if gate_array.shape[0] == 0: gate_array = np.zeros((Param.max_gate_num, Param.gate_feat_in_num))
            if obs_array.shape[0] == 0: obs_array = np.zeros((Param.max_obs_num, Param.obs_feat_in_num))
            num_obs.append(obs_array.shape[0]); num_gate.append(gate_array.shape[0])
            # ---- PAD ----
            gate_array = np.pad(gate_array, ((0, Param.max_gate_num - gate_array.shape[0]), (0, 0)))
            obs_array = np.pad(obs_array, ((0, Param.max_obs_num - obs_array.shape[0]), (0, 0)))
            res_obs.append(obs_array)
            res_gate.append(gate_array)
        res_obs = np.array(res_obs, dtype=float); res_gate = np.array(res_gate, dtype=float)
        num_obs = np.array(num_obs, dtype=float); num_gate = np.array(num_gate, dtype=float)
        num_room = res_obs.shape[0]
        res_obs = np.pad(res_obs, ((0, Param.max_room_num - res_obs.shape[0]), (0, 0), (0, 0)))
        res_gate = np.pad(res_gate, ((0, Param.max_room_num - res_gate.shape[0]), (0, 0), (0, 0)))
        num_obs = np.pad(num_obs, (0, Param.max_room_num - num_obs.shape[0]))
        num_gate = np.pad(num_gate, (0, Param.max_room_num - num_gate.shape[0]))
        return num_room, num_obs, num_gate, res_obs, res_gate
    '''
    def guess_subroom_env_info(self, env_info):
        '''
        NOTE assume agentA is in the center of this room
        TODO not sure if obs and gate should be dealt with separately
        TODO is it necessary to add type in vectors ?
        :param env_info:
        :return:
        '''
        # res = {}
        res_obs = []; res_gate = []
        num_obs = []; num_gate = []
        # for room_str, room_info in env_info.items():
        for room_id in range(len(env_info.keys())):  # NOTE make sure that the room is ordered
            room_str = str(room_id); room_info = env_info[room_str]
            gates_info = room_info["gates"]
            obs_info = room_info["obstacles"]
            room_pos = room_info["pos"]
            room_pos = [int(i) for i in room_pos]
            agentA_pos = ((room_pos[0] + room_pos[1])//2 - room_pos[0], (room_pos[2] + room_pos[3])//2 - room_pos[2])
            temp_obs=[]; temp_gate=[]
            for sub_id in range(4):
                # room_vectors = []
                gate_vectors = []; obs_vectors = []
                # ---- gate ----
                # for gate_str, cur_gate_info in gates_info.items():
                for gate_id in range(len(gates_info.keys())):  # NOTE make sure that the gate is ordered
                    gate_str = str(gate_id); cur_gate_info = gates_info[gate_str]
                    cur_vector,sub = self._get_gate_vector(agentA_pos, cur_gate_info)
                    if sub==sub_id:
                        gate_vectors.append(cur_vector)
                # ---- obs ----
                # for obs_str, cur_obs_info in obs_info.items():
                for obs_id in range(len(obs_info.keys())):  # NOTE make sure that the obs is ordered
                    obs_str = str(obs_id); cur_obs_info = obs_info[obs_str]
                    cur_vector,sub = self._get_obs_vector(agentA_pos, cur_obs_info)
                    if sub==sub_id:
                        obs_vectors.append(cur_vector)
                gate_array = np.array(gate_vectors, dtype=float)
                obs_array = np.array(obs_vectors, dtype=float)
                if gate_array.shape[0] == 0: gate_array = np.zeros((Param.max_subgate_num, Param.gate_feat_in_num))
                if obs_array.shape[0] == 0: obs_array = np.zeros((Param.max_subobs_num, Param.obs_feat_in_num))
                #num_obs.append(obs_array.shape[0]); num_gate.append(gate_array.shape[0])
                # ---- PAD ----
                gate_array = np.pad(gate_array, ((0, Param.max_subgate_num - gate_array.shape[0]), (0, 0)))
                obs_array = np.pad(obs_array, ((0, Param.max_subobs_num - obs_array.shape[0]), (0, 0)))
                temp_obs.append(obs_array)
                temp_gate.append(gate_array)
            #三维:4*max*feat_in
            temp_gate=np.array(temp_gate, dtype=float)
            temp_obs=np.array(temp_obs, dtype=float)           
            res_obs.append(temp_obs)
            res_gate.append(temp_gate)
        #四维：9*4*max*feat_in
        res_obs = np.array(res_obs, dtype=float); res_gate = np.array(res_gate, dtype=float)
        #num_obs = np.array(num_obs, dtype=float); num_gate = np.array(num_gate, dtype=float)
        num_room = res_obs.shape[0]
        assert num_room==Param.max_room_num
        #res_obs = np.pad(res_obs, ((0, Param.max_room_num - res_obs.shape[0]), (0, 0), (0, 0)))
        #res_gate = np.pad(res_gate, ((0, Param.max_room_num - res_gate.shape[0]), (0, 0), (0, 0)))
        #num_obs = np.pad(num_obs, (0, Param.max_room_num - num_obs.shape[0]))
        #num_gate = np.pad(num_gate, (0, Param.max_room_num - num_gate.shape[0]))
        return num_room, num_obs, num_gate, res_obs, res_gate
    '''
    def _get_gate_vector(self, agentA_pos, cur_gate_info):
        # obj_type = [1, 0]  # gate
        # cur_shape = [0 for _ in range(len(self.shapes))]  # gate has no prop of shape or color
        # cur_color = [0 for _ in range(len(self.colors))]
        # pos compared with agentA, [up, down, left, right].
        '''
        cur_pos = [0 for _ in range(len(self.directions))]
        # 0,1代表上下，2，3代表左右
        cur_dir_index = self.direction_mapping[cur_gate_info["direction"]]
        cur_pos[cur_dir_index] = 1
        #0代表左右，1代表上下
        oppo_axis = 1 - cur_dir_index // 2
        if agentA_pos[oppo_axis] < int(cur_gate_info["pos"]):
            cur_pos[2 * oppo_axis + 1] = 1
        elif agentA_pos[oppo_axis] > int(cur_gate_info["pos"]):
            cur_pos[2 * oppo_axis] = 1
        # if equal then do nothing
        '''
        #物体相对于agent ，位于agent的哪边
        cur_pos = [0 for _ in range(2)]
        #0,1:up,down  2,3:left,right
        cur_dir_index = self.direction_mapping[cur_gate_info["direction"]]
        if cur_dir_index==0:
            cur_pos[0]=-1
        elif cur_dir_index==1:
            cur_pos[0]=1
        elif cur_dir_index==2:
            cur_pos[1]=-1
        elif cur_dir_index==3:
            cur_pos[1]=1
        #1:up,down  0:left,right
        oppo_axis=1-cur_dir_index//2
        if int(cur_gate_info["pos"]) < agentA_pos[oppo_axis] :
            cur_pos[oppo_axis] = -1
        elif int(cur_gate_info["pos"]) > agentA_pos[oppo_axis] :
            cur_pos[oppo_axis] = 1
        #add color of doors
        cur_color = [self.color_mapping[cur_gate_info["color"]]]
        cur_vector = cur_pos+cur_color
        #add four subrooms
        if cur_pos[0]==1:
            if cur_pos[1]==1 :
                sub=2
            else:
                sub=3
        elif cur_pos[1]==1:
            sub=1
        else:
            sub=0
        return cur_vector,sub
        

    def _get_obs_vector(self, agentA_pos, cur_obs_info):
        # obj_type = [0, 1]  # obs
        #cur_shape = [0 for _ in range(len(self.shapes))]
        #cur_shape[self.shape_mapping[cur_obs_info["shape"]]] = 1
        cur_shape=[self.shape_mapping[cur_obs_info["shape"]]]
        #cur_color = [0 for _ in range(len(self.colors))]
        #cur_color[self.color_mapping[cur_obs_info["color"]]] = 1
        cur_color=[self.color_mapping[cur_obs_info["color"]]]
        # pos compared with agentA, [up, down, left, right]
        #cur_pos = [0 for _ in range(len(self.directions))]
        #if int(cur_obs_info["pos"][0]) < agentA_pos[0]: cur_pos[0] = 1
        #elif int(cur_obs_info["pos"][0]) > agentA_pos[0]: cur_pos[1] = 1
        #if int(cur_obs_info["pos"][1]) < agentA_pos[1]: cur_pos[2] = 1
        #elif int(cur_obs_info["pos"][1]) > agentA_pos[1]: cur_pos[3] = 1
        #-1:up,left 1:down,right 0:same
        cur_pos = [0 for _ in range(2)]
        if int(cur_obs_info["pos"][0]) < agentA_pos[0]: cur_pos[0] = -1
        elif int(cur_obs_info["pos"][0]) > agentA_pos[0]: cur_pos[0] = 1
        if int(cur_obs_info["pos"][1]) < agentA_pos[1]: cur_pos[1] = -1
        elif int(cur_obs_info["pos"][1]) > agentA_pos[1]: cur_pos[1] = 1
        cur_vector = cur_shape + cur_color + cur_pos
        #add four subrooms
        if cur_pos[0]==1:
            if cur_pos[1]==1 :
                sub=2
            else:
                sub=3
        elif cur_pos[1]==1:
            sub=1
        else:
            sub=0
        return cur_vector,sub

"""
    