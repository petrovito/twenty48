

if __name__ == '__main__':
    from subprocess import Popen, PIPE
    from datetime import datetime
    import os
    import qlparams

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M-")

    param_name = "num_epochs"
    values = [20, 30, 50]

    params = dict(qlparams.params)
    exp_name = "lowest"
    dir_name = "./logs/"+timestamp+param_name+"/"
    os.mkdir(dir_name)
    f = open(dir_name+"desc", 'w')
    f.write(exp_name+'\n'+param_name+": "+str(values)+"\n"+str(params)+"\n")
    f.close()
    for val in values:
        Popen(["python ql.py " + param_name + " " + exp_name + " " + dir_name + " " + str(val)], stdout=PIPE,
                   bufsize=1, close_fds=True,
                   universal_newlines=True, shell=True)
