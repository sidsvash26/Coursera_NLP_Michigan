class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        #Define words        
        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer[0]
                
        
        #Pre-condition:
        
        #wi shouldn't have been a dependent of some other word 
        
        if any(z for x,y,z in conf.arcs if z == idx_wi):
            return -1
            
        #Also, wi should not be root
        if idx_wi == 0 :
            return -1        
        
        #Transition
        conf.stack = conf.stack[0:len(conf.stack)-1]    #stack is reduced by last word
        conf.arcs.append((idx_wj, relation, idx_wi))   #relation added to arc
        # debug line print 'left_arc'
        
        return conf   
        #raise NotImplementedError('Please implement left_arc!')
        #return -1

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))
        # debug line print 'right_arc'
        return conf
        
    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        #Pre-condition
        idx_wi = conf.stack[-1]
        
        #if not a dependent already, reduce shouldn't happen
        if any(z for x,y,z in conf.arcs if z == idx_wi):  
           conf.stack.pop()  
         # debug line   print 'reduced'
           return conf
           
        else:
           return -1
        
        #raise NotImplementedError('Please implement reduce!')
        

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        
        idx_wj = conf.buffer.pop(0)
        
        conf.stack.append(idx_wj)
        # debug line print 'shifted'
        
        return conf
