raiddir = /raid/embe/adapointr
all: docker
docker:
	docker build -t ${USER}/adapointr:latest .
clean:
	docker run --rm -v $(CURDIR):/pointr busybox:latest find /pointr -user root -exec rm -rf {} +
run:
	srun --pty --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:1 --job-name=AdaPoinTr bash -ic \
	"nvidia-docker run -it --rm -p 8931:22 -e WANDB_API_KEY=$(shell awk '/api.wandb.ai/ {f=1} f && /password/ {print $$2;f=0}' ~/.netrc) -v $(CURDIR):/pointr -v /raid/ska/illustris:/cluster/data/ska/illustris ${USER}/adapointr:latest bash"
train:
	srun --pty --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --job-name=AdaPoinTr bash -ic \
	"nvidia-docker run -it --rm -p 8931:22 -e WANDB_API_KEY=$(shell awk '/api.wandb.ai/ {f=1} f && /password/ {print $$2;f=0}' ~/.netrc) -v $(CURDIR):/pointr -v /raid/ska/illustris:/cluster/data/ska/illustris ${USER}/adapointr:latest python main.py --config cfgs/Illustris/AdaPoinTr.yaml --val_freq 50 --num_queries=512 --opt=AdamW --opt_lr=2.0e-06 --opt_wd=0.03 --sched=LambdaLR --total_bs=32 --exp_name dm2gas_2000_ng_s_l2"
strain:
	WANDB__EXECUTABLE=sys.executable; \
	srun --pty --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:1 --export=WANDB__EXECUTABLE --job-name=AdaPoinTr \
	python main.py --config cfgs/Illustris/AdaPoinTr.yaml --exp_name train_baseline_conda
export:
	docker run -it --rm -v $(CURDIR):/skais -v /raid/ska/illustris:/cluster/data/ska/illustris -v /cluster/data/ska/illustris:/skais/data ${USER}/base:latest bash
stall:
	srun --pty --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --job-name=AdaPoinTr bash -ic \
	"nvidia-docker run -it --rm -e WANDB_API_KEY=$(shell awk '/api.wandb.ai/ {f=1} f && /password/ {print $$2;f=0}' ~/.netrc) -v $(CURDIR):/pointr -v /raid/ska/illustris:/cluster/data/ska/illustris ${USER}/adapointr:latest bash"
subset:
	srun --pty --ntasks=1 --cpus-per-task=8 --mem=128G --job-name=AdaPoinTr bash -ic \
	"nvidia-docker run -it --rm -e WANDB_API_KEY=$(shell awk '/api.wandb.ai/ {f=1} f && /password/ {print $$2;f=0}' ~/.netrc) -v $(CURDIR):/pointr -v /raid/ska/illustris:/cluster/data/ska/illustris ${USER}/adapointr:latest python h5subset.py cfgs/Illustris/AdaPoinTr.yaml"
agent:
	srun --pty --ntasks=1 --cpus-per-task=8 --mem=128G --job-name=AdaPoinTr bash -ic \
	"nvidia-docker run -it --rm -e WANDB_API_KEY=$(shell awk '/api.wandb.ai/ {f=1} f && /password/ {print $$2;f=0}' ~/.netrc) -v $(CURDIR):/pointr -v /raid/ska/illustris:/cluster/data/ska/illustris ${USER}/adapointr:latest wandb agent raember/MT/vqtr3qg2"
plot:
	python tools/point_collector.py --pc - cfgs/Illustris/AdaPoinTr.yaml ckpt-e1250_ng_s_l2.pth
aputrain:
	python main.py --config cfgs/Illustris/AdaPoinTr.yaml --num_workers 1 --exp_name baseline
inference:
	python tools/point_collector.py --pc - cfgs/Illustris/AdaPoinTr.yaml ckpt-e930_ng.pth
raid:
	mkdir -p "$(raiddir)"
	cp main.py "$(raiddir)"
	cp Makefile "$(raiddir)"
	rsync -a cfgs "$(raiddir)/" --exclude '**/__pycache__'
	rsync -a data "$(raiddir)/" --exclude '**/__pycache__'
	rsync -a datasets "$(raiddir)/" --exclude '**/__pycache__'
	rsync -a extensions "$(raiddir)/" --exclude '**/__pycache__'
	rsync -a models "$(raiddir)/" --exclude '**/__pycache__'
	rsync -a tools "$(raiddir)/" --exclude '**/__pycache__'
	rsync -a utils "$(raiddir)/" --exclude '**/__pycache__'
