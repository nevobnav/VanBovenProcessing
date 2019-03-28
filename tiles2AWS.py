import paramiko
import os
import shutil


print('hello world')

local_path = r"C:\Users\VanBoven\Documents\GitHub\VanBovenProcessing\testfolder"
pem_path= r"C:\Users\VanBoven\Documents\SSH\VanBovenAdmin.pem"
target_path = "/home/ubuntu/media/data/"
#Create zipfile of local_path
shutil.make_archive(os.path.basename(local_path), 'zip', local_path)

connect = False

if connect:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ec2-52-29-220-114.eu-central-1.compute.amazonaws.com',
                username='ubuntu',
                key_filename=pem_path)
    sftp = ssh.open_sftp()

    remote_base_folder = target_path + os.path.basename(local_path)
    sftp.mkdir(remote_base_folder)


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
