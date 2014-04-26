%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = det_rpc(ins, tas_params)

ni=length(ins.name);
pl = ins.score;
np=0;
ndets = 0;
dets = [];
recs = [];
masks = [];

% Allocate
det_counts = 0;
for i = 1:ni
    cands = ins.dets{i};
    det_counts = det_counts + size(cands,1);
    masks{i} = read_mask(ins.gt_filename{i}, tas_params.scale);
end
dets(det_counts).imgnum = -1;
dets(det_counts).bbox   = [];
dets(det_counts).confidence = -inf;

% Copy
for i = 1:ni
    recs(i).objects = [];
    
    % DETECTIONS
    cands = ins.dets{i};
    for n = 1:size(cands,1)
        %Check detection confidence is high enough
        if pl{i}(n) >= tas_params.cand_threshold
            ndets = ndets + 1;
            dets(ndets).imgnum = i;
            dets(ndets).bbox = cands(n,:);
            dets(ndets).confidence = pl{i}(n);
        end
    end
end

nd = ndets;

if(~isempty(dets))
    [sc,si]=sort(-[dets(:).confidence]);
    dets=dets(:,si);
else
    si = [];
end

tp=zeros(nd,1);
fp=zeros(nd,1);
for d=1:nd
    i=dets(d).imgnum;
    bb=dets(d).bbox;

    ov = detection_overlap(bb, masks{i});
    if ov > tas_params.truth_threshold
            tp(d)=1;
    else
            fp(d)=1;
    end
end

tp=cumsum(tp);
tp(end)
fp=cumsum(fp);
fp(end)

end
