class PipelineStatus:
    def __init__(self, on_hold, data=None):
        self.on_hold = on_hold
        self.data = data

class PipelineID:
    def __init__(self,rank,number=0):
        self.number = number
        self.rank = rank

class PipelineManager:
    def __init__(self,communicator):
        self._IDs = {}
        self._particles_per_rank = {}
        self._elements = {}
        self._pending_requests = {}
        self._last_request_turn = {}

        self._communicator = communicator
        #self._max_tag = self._comm.Get_attr(MPI.TAG_UB) # 8388607 with OpenMPI on HPC photon

    def add_particles(self,particles_name,rank):
        if rank in self._particles_per_rank.keys():
            pipeline_number = len(self._particles_per_rank[rank])
        else:
            pipeline_number = 0
            self._particles_per_rank[rank] = []
        pipeline_ID = PipelineID(rank,pipeline_number)
        self._IDs[particles_name] = pipeline_ID
        self._particles_per_rank[rank].append(particles_name)

    def get_particles_ID(self,particles_name):
        return self._IDs[particles_name]

    def get_particles_rank(self,particles_name):
        return self._IDs[particles_name].rank

    def add_element(self,element_name):
        self._elements[element_name] = len(self._elements)

    def get_message_tag(self,element_name,sender_name,reciever_name):
        tag = self._elements[element_name] + len(self._elements)*self._IDs[sender_name].number+len(self._elements)*len(self._IDs)*self._IDs[reciever_name].number
        #tag = self.pipeline_number+self.pipeline_max_size*sender.number+self.pipeline_max_size*self.pipeline_max_particles_per_rank*reciever.number
        #if tag > self._max_tag:
        #    print(f'PyPLINEDElement WARNING {self.name}: MPI message tag {tag} is larger than max ({self._max_tag})')
        return tag

    def is_ready_to_send(self,element_name,sender_name,reciever_name,turn):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name)
        if tag not in self._last_request_turn.keys():
            return True
        if turn <= self._last_request_turn[tag]:
            return False
        if not self._pending_requests[tag].Test():
            return False
        return True

    def send_message(self,send_buffer,element_name,sender_name,reciever_name,turn):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name)
        self._last_request_turn[tag] = turn
        self._pending_requests[tag] = self._communicator.Issend(send_buffer,dest=self.get_particles_rank(reciever_name),tag=tag)

    def is_ready_to_recieve(self,element_name,sender_name,reciever_name):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name)
        return self._communicator.Iprobe(source=self.get_particles_rank(sender_name), tag=tag)

    def recieve_message(self,recieve_buffer,element_name,sender_name,reciever_name):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name)
        self._communicator.Recv(recieve_buffer,source=self.get_particles_rank(sender_name),tag=tag)

class PipelineBranch:
    def __init__(self, tracker, particles):
        self.tracker = tracker
        self.particles = particles
        self.pipeline_status = None

        self.tracker.enable_pipeline_hold = True

class PipelineMultiTracker:

    def __init__(self, branches):
        self.branches = branches

    def track(self, **kwargs):
        for branch in self.branches:
            branch.pipeline_status = branch.tracker.track(
                 branch.particles, **kwargs)

        while True:
            prrrrrr

