import paramiko
import os,time
import shutil
from vanbovendatabase.postgres_lib import *

#### INPUT VARIABLES #######
local_path = r"C:\Tiles4\Lefeber\20190326"


### CONFIG ###
with open('postgis_config.json') as config_file:
    config = json.load(config_file)
DB_NAME = config['NAME']
DB_USER = config['DB_USER']
DB_PASSWORD = config['DB_PASSWORD']
DB_IP = config['DB_IP']

pem_path= r"C:\Users\VanBoven\Documents\SSH\VanBovenAdmin.pem"
remote_path = "/home/ubuntu/media/data/"
print('Config read')
### PROGRAM ###

#Get customer_id from plotname and render the full remote path
plotname = os.path.basename(os.path.split(local_path)[0])
print(plotname)
con,meta = connect(DB_USER, DB_PASSWORD, DB_NAME, host=DB_IP)
customer_pk = get_plot_customer(plotname, meta, con)
customer_id = get_customer_id(customer_pk,meta,con)
con.connect().close()
print('Customer id: ',customer_id)
full_remote_path = (remote_path + customer_id + '/' + plotname + '/').replace(' ','\ ')

print('Remote path: ', full_remote_path)

#Create zipfile of tile folder
if False:
    print('zipping...')
    shutil.make_archive(local_path, 'zip', local_path)
    print('finished zipping')
zipname = os.path.basename(local_path)+'.zip'
full_remote_zip_path = (remote_path + customer_id + '/' + plotname + '/' + zipname).replace(' ','\ ')
fullname = local_path + '.zip'
filesize = os.path.getsize(fullname)

#render unzip command
command_unzip = 'unzip ' + full_remote_zip_path
command_removezip = 'rm ' + full_remote_zip_path
print(command_removezip)

connection = True

#We should zip the '20190322' folder, which should end up in the customer_id/plotname folder

if connection:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ec2-52-29-220-114.eu-central-1.compute.amazonaws.com',
                username='ubuntu',
                key_filename=pem_path)
    sftp = ssh.open_sftp()

    ## Create required path
    #Try to make the customer path
    try:
        sftp.mkdir(remote_path + customer_id)
    except:
        pass #This means the customer_id path already exists.
    try:
        sftp.mkdir(remote_path + customer_id + '/' + plotname)
    except:
        pass #This means the plotname path already exists
    try:
        sftp.mkdir(remote_path + customer_id + '/' + plotname + '/' + os.path.basename(local_path ))
    except:
        pass #This would be odd... scan already exists

    #remote_attr = sftp.put(fullname,full_remote_zip_path)
    #if remote_attr.st_size == filesize:
    #    print('Succesfull upload! {}/{}'.format(remote_attr.st_size, filesize))
    #else:
    #    print('Upload failed: {}/{}'.format(remote_attr.st_size, filesize))


    # Unzip .zip folder
    #(stdin, stdout, stderr) = ssh.exec_command(command_unzip)
    #print(stdin.read())

    #(stdin, stdout, stderr) = ssh.exec_command(command_removezip)
    #print(stdin)
    print(command_unzip)
    sleeptime = 0
    stdin, stdout, stderr = ssh.exec_command(command_unzip)
    stdout.flush()
    nbytes = 0
    while True:
        print(stdout.readline())
        print('here we are')
        if stdout.channel.exit_status_ready():
            break

    while not stdout.channel.eof_received:
        time.sleep(1)
        sleeptime += 1
        print(sleeptime)
        if sleeptime > 30:
            stdout.channel.close()
            break
    stdout.read()





    #stdout_.channel.recv_exit_status()
    #lines = stdout_.readlines()
    #for line in lines:
    #    print(line)


    sftp.close()




# def put_all(ssh,localpath,remotepath):
#         #  recursively upload a full directory
#         os.chdir(os.path.split(localpath)[0])
#         parent=os.path.split(localpath)[1]
#         for walker in os.walk(parent):
#             try:
#                 ssh.sftp.mkdir(os.path.join(remotepath,walker[0]))
#             except:
#                 pass
#             for file in walker[2]:
#                 ssh.put(os.path.join(walker[0],file),os.path.join(remotepath,walker[0],file))
# for root, dirs, files in os.walk(local_path, topdown=True):
#     for name in dirs:
#         dirname = name
#         full_dirname = (os.path.join(root, name))
#         dir_tail = full_dirname.replace(local_path,'')
#         remote_path = (remote_base_folder+dir_tail).replace('\\','/')
#         if connect:
#             sftp.mkdir(remote_path)
#
#     for name in files:
#         filename = name
#         full_name = os.path.join(root, name)
#         dir_tail = full_name.replace(local_path,'')
#         filesize = os.path.getsize(full_name)
#         remote_path = (remote_base_folder+dir_tail).replace('\\','/')
#         print('Remote path:',remote_path)
#         print('Full name:', full_name)
#         if connect:
#             remote_attr = sftp.put(full_name,remote_path)
#         succes = (remote_attr.st_size == filesize)
#         print(succes)
