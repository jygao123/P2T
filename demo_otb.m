clear;
clc;
addpath(genpath('utils'));

opts = genOpts();
seqs = importdata(opts.seqList);

times = 1;
for i = 1 : length(seqs)
    opts.roiPath  = fullfile(opts.roiDir, ['roidb_' seqs{i} '.mat']);
    opts.train.seqName = seqs{i};
    
    if exist(opts.roiPath,'file')
        load(opts.roiPath) ;
    else
        roidb = setup_data_re(opts.dataset, seqs{i}, opts.sampling);
        save(opts.roiPath, 'roidb') ;
    end
    model_path = ['./model/' seqs{i} '.caffemodel'];
    if ~exist(model_path,'file')
        pretrain_re( roidb, opts.train, seqs{i} );
    end
    opts.tracking.seqName = seqs{i};
    res_path=['./results/' num2str(times) '/' opts.tracking.seqName '/'];
    opts.tracking.res_path = res_path;
    if ~exist(res_path,'dir')
        mkdir(res_path);
    end
    
    getNet(opts.tracking);
    
    conf = genConfig(opts.dataset,opts.tracking.seqName);
    gts = conf.gt;
    
    result = tracking_run_fe(opts.tracking, conf.imgList, conf.gt(1,:), true, times, res_path,i);
end
