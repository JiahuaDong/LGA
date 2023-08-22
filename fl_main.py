import copy
import os.path as osp
import time

import torch.cuda
from timm.models.resnet import resnet10t
from timm.models.vgg import vgg16,vgg11,vgg13
from timm.models.layers import ClassifierHead
from LGA import LGA
from ProxyServer import *
from ResNet import resnet18_cbam
from mini_imagenet import *
from option import args_parser
from tiny_imagenet import *
import warnings
warnings.filterwarnings("ignore")
args = args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
args.device=0

feature_extractor = resnet18_cbam()  # extractor
num_clients = args.num_clients
old_client_0 = []
old_client_1 = [i for i in range(args.num_clients)]
new_client = []
models = []
total_class=(args.task_size*args.epochs_global)/args.tasks_global
total_class=int(total_class)

# seed settings
setup_seed(args.seed)

# random crop and padding 
train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
									  transforms.RandomHorizontalFlip(p=0.5),
									  transforms.ColorJitter(brightness=0.24705882352941178),
									  transforms.ToTensor(),
									  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),
									 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

if args.dataset == 'cifar100':
	train_dataset = iCIFAR100(args.dataset_path, transform=train_transform, download=True)
	test_dataset = iCIFAR100(args.dataset_path, test_transform=test_transform, train=False, download=True)
	model_g = network(args.numclass, feature_extractor, 4)
	if args.encode=='resnet18':
		encode_model = network(total_class, feature_extractor, 4)
	elif args.encode=='vgg11':
		encode_model = vgg11()
		num=encode_model.num_features
		encode_model.head=ClassifierHead(num,total_class,pool_type='avg',drop_rate=0.)
	else:
		encode_model = LeNet(5, 2, num_classes=total_class)
		encode_model.apply(weights_init)
	num_img=10
elif args.dataset == 'tiny_imagenet':
	train_dataset = Tiny_Imagenet(args.dataset_path, train_transform=train_transform, test_transform=test_transform)
	train_dataset.get_data()
	test_dataset = train_dataset
	model_g = network(args.numclass, feature_extractor, 8)
	if args.encode=='resnet18':
		backbone=copy.deepcopy(feature_extractor)
		encode_model = network(total_class, backbone, 8)
	elif args.encode=='resnet10':
		encode_model = vgg11()
		num=encode_model.num_featuresd
		encode_model.head=ClassifierHead(num,total_class,pool_type='avg',drop_rate=0.)
	elif args.encode == 'vgg16':
		encode_model = vgg16()
		num=encode_model.num_features
		encode_model.head=ClassifierHead(num,total_class,pool_type='avg',drop_rate=0.)
	else:
		encode_model = LeNet(5, 4, num_classes=total_class)
		encode_model.apply(weights_init)
	num_img = 9
else:
	train_dataset = Mini_Imagenet(args.dataset_path, train_transform=train_transform, test_transform=test_transform)
	train_dataset.get_data()
	test_dataset = train_dataset
	model_g = network(args.numclass, feature_extractor, 11)
	encode_model = LeNet(3, 4, num_classes=total_class)
	encode_model.apply(weights_init)
	num_img = 10
model_g = model_to_device(model_g, False, args.device)
model_old = None
# encode_model.apply(weights_init)

# los log	
dir = osp.join('./los_log')
if not osp.exists(dir):
	os.system('mkdir -p ' + dir)
if not osp.exists(dir):
	os.mkdir(dir)
	

for i in range(250):  # model_client
	model_temp = LGA(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
							args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model,args)
	models.append(model_temp)

# the proxy server
proxy_server = proxyServer(args.device, args.learning_rate, args.numclass, feature_extractor, encode_model,
						   train_transform,args)

# training log
output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))
if not osp.exists(output_dir):
	os.system('mkdir -p ' + output_dir)
if not osp.exists(output_dir):
	os.mkdir(output_dir)


# model save
model_local_dir = osp.join('./model', 'model_local')
if not osp.exists(model_local_dir):
	os.system('mkdir -p ' + model_local_dir)
if not osp.exists(model_local_dir):
	os.mkdir(model_local_dir)

model_global_dir = osp.join('./model', 'model_global')
if not osp.exists(model_global_dir):
	os.system('mkdir -p ' + model_global_dir)
if not osp.exists(model_global_dir):
	os.mkdir(model_global_dir)

out_file = open(osp.join(output_dir, args.method+ '_'+str(args.dataset)+ '_task_size_' +str(args.task_size) + str(args.memory_size)+ str(int(time.time())) + '.txt'), 'w')
log_str = 'method_{}, task_size_{}, learning_rate_{}'.format(args.method, args.task_size, args.learning_rate)
out_file.write(log_str + '\n')
out_file.flush()


classes_learned = args.task_size
old_task_id = -1
for ep_g in range(args.epochs_global):
	pool_grad = [] #
	task_id = ep_g // args.tasks_global
	model_old = proxy_server.model_back()

	if task_id != old_task_id and old_task_id != -1:
		overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
		new_client = [i for i in range(overall_client, overall_client + 10)] 
		old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
		old_client_0 = [i for i in range(overall_client) if i not in old_client_1] 
		print(old_client_0)

	if task_id != old_task_id and old_task_id != -1:
		classes_learned += args.task_size
		model_g.Incremental_learning(classes_learned)
		model_g = model_to_device(model_g, False, args.device)

	if ep_g==args.tasks_global:
		dataset=copy.deepcopy(train_dataset)
		model_g.compute_radius(model_g, dataset, args.task_size, args.device, num_img)
		proxy_server.best_perf = 0

	print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

	w_local = []

	clients_index = random.sample(range(num_clients), args.local_clients)
	print('select part of clients to conduct local training')
	print(clients_index)
	for c in clients_index:
		local_model, proto_grad = local_train(models, c, model_g, task_id, model_old, ep_g, old_client_0)
		w_local.append(local_model)
		if proto_grad is not None:
			for grad_i in proto_grad:
				pool_grad.append(grad_i)

	# every participant save their current training data as exemplar set
	print('every participant start updating their exemplar set and old model...')
	participant_exemplar_storing(models, num_clients, model_g, old_client_0, task_id, clients_index)
	print('updating finishes')

	print('federated aggregation...')
	w_g_new = FedAvg(w_local)
	w_g_last = copy.deepcopy(model_g.state_dict())

	model_g.load_state_dict(w_g_new)
	proxy_server.ep_g=ep_g
	proxy_server.model = copy.deepcopy(model_g)
	proxy_server.dataloader(pool_grad, models[0])
	cur=proxy_server.cur_perf
	
	acc_global= model_global_eval(model_g, test_dataset, task_id, args.task_size, args.device)
	log_str = 'Task: {}, Round: {} Accuracy = {:.2f}%  radius={:.2f}% cur={:.2f}%'. \
		format(task_id, ep_g, acc_global, model_g.radius, cur)
	out_file.write(log_str + time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(time.time())) + '\n')
	out_file.flush()
	print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, acc_global))
	old_task_id = task_id
