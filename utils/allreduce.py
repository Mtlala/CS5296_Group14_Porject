from torch import distributed as dist

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        """ using all_reduce """
        # dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        # param.grad.data /= size

        """ using gather and scatter """
        # group = dist.new_group(list(range(size)))
        # dist.gather(tensor=param.grad.data, dst=0, gather_list=group, group=group)
        # if rank == 0:
        #     param.grad.data /= size
        #
        # dist.scatter(tensor=param.grad.data, src=0, scatter_list=group, group=group)

        """ using ring-reduce """
        ringreduce(param.grad.data, param.grad.data)
        param.grad.data /= size

""" Implementation of a ring-reduce with addition. """
def ringreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]