import matchPatches
import threading
import time

# TODO: a queue of all jobs
myJobs = []
mylock = threading.Lock()

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter = None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        start_time = time.time()
        print "Starting " + self.name + "at ", start_time
        dispatch_job(self.name)
        print "Exiting " + self.name + 'finish matching; time spent:', time.time() - start_time
        # dispatch_a_new_job()

def dispatch_job(threadName):
    folder_suffix = "_HOG_Jensen_Shannon_Divergence"
    if(threadName == "populate_testset_illuminance1"):
        matchPatches.populate_testset_illuminance1(folder_suffix)
    elif(threadName == "populate_testset_illuminance2"):
        matchPatches.populate_testset_illuminance2(folder_suffix)
    elif(threadName == "populate_testset_rotation1"):
        matchPatches.populate_testset_rotation1(folder_suffix)
    elif(threadName == "populate_testset_rotation2"):
        matchPatches.populate_testset_rotation2(folder_suffix)
    elif(threadName == "populate_testset4"):
        matchPatches.populate_testset4(folder_suffix)
    elif(threadName == "populate_testset7"):
        matchPatches.populate_testset7(folder_suffix)    


def dispatch_a_new_job():
    if(threading.activeCount() >= 4):
        return
    mylock.acquire()
    if(len(myJobs) > 0):
        # retrieve the front thread from the myJobs [] and start it
        # remove that thread from the myJobs [] list
        newThread = myJobs[0]
        newThread.start()
        myJobs = myJobs.pop(0)
    mylock.release()
    return

def main():

    # Create new threads
    thread1 = myThread(1, "populate_testset_illuminance1")
    thread2 = myThread(2, "populate_testset_illuminance2")
    thread3 = myThread(3, "populate_testset_rotation1")
    thread4 = myThread(4, "populate_testset_rotation2")
    thread5 = myThread(5, "populate_testset4")
    thread6 = myThread(6, "populate_testset7")

    # Start new Threads
    thread1.start()
    thread2.start()
    # thread3.start()
    thread4.start()
    # thread5.start()
    thread6.start()

    return

if __name__ == '__main__':
    main()