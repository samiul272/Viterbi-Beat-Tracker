function [mainscore, backupscores]= beatEvaluator2(detections,annotations,metricalOptions)

%  function [mainscore, backupscores]= beatEvaluator2(detections,annotations)
%   
%  Description:
%  Calculate the continuity based accuracy values as used in (Hainsworth, 2004) and (Klapuri et al, 2006)
%  UPDATE to beatEvaluator.m which now allows the set of allowed
%  metricals level to be specified as an input parameter, and fix minor issue with minBeatTime parameter.
%   
%   
%  Inputs: 
%  detections - sequence of estimated beat times (in seconds)
%  annotations - sequence of ground truth beat annotations (in seconds)
%  metricalOptions - a matrix of numVariations (rows) by three columns with the
%  following information: [startingAnnotation factor offbeat]
%  e.g. the default condition
%  [1 1 0] -> unchanged
%  [1 1 1] -> offbeat
%  [1 2 0] -> double time
%  [1 0.5 0] -> half time start first annotation
%  [2 0.5 0] -> half time start second annotation
%
%  implement as:     
%  metricalOptions = [1 1 0; 1 1 1; 1 2 0; 1 0.5 0; 2 0.5 0];
%
%  other options might include a piece in 5/4:
%  [1 1 0] -> unchanged
%  [1 2 0] -> double time
%  [2 1/5 0] -> every fifth annotation starting at the second
%   
%  implement as:     
%  metricalOptions = [1 1 0; 1 2 0; 2 1/5 0];
%
%  Ouputs: 
%  mainscore - continuity not required at allowed metrical levels (amlT)
%  backupscores - the remaining continuity conditions, to be used for
%  tiebreaking (amlc, cmlt, cmlc). 
%
%  References:
%
%  S. Hainsworth, "Techniques for the automated analysis of musical audio,"
%  Ph.D. dissertation, Department of Engineering, Cambridge University,
%  2004.
%
%  A. P. Klapuri, A. Eronen, and J. Astola, "Analysis of the meter of
%  acoustic musical signals," IEEE Transactions on Audio, Speech and
%  Language Processing, vol. 14, no. 1, pp. 342-355, 2006.
%
%  M. E. P. Davies, N. Degara and M. D. Plumbley. "Evaluation Methods for
%  Musical Audio Beat Tracking Algorithms," Technical Report C4DM-TR-09-06,
%  Queen Mary University of London, Centre for Digital Music, 8 October
%  2009.
%
%  S. Boeck, F. Korzeniowski, J. Schluter, F. Krebs, G. Widmer "madmom: a
%  new Python Audio and Music Signal Processing Library,"
%  https://arxiv.org/pdf/1605.07008.pdf
%
%
%  This provides near identical output to Sebastian Boeck's madmom evaluation 
%  code in python: (expect that this implementation allows a 5s start up
%  period): https://github.com/CPJKU/madmom
%
%
%  (c) 2016 Matthew Davies, INESC TEC

% set up the parameters
% start up period - this is the default setting which is updated below to handle special cases were an 
startUpTime = 5;
% size of tolerance window for beat phase in continuity based evaluation
phase_tolerance = 0.175;
% size of tolerance window for beat period in continuity based evaluation
tempo_tolerance = 0.175;


if nargin<3
    % use default metrical options  
    metricalOptions = [1 1 0; 1 1 1; 1 2 0; 1 0.5 0; 2 0.5 0];
    % metricalOptions(1,:) -> unchanged
    % metricalOptions(2,:) -> offbeat
    % metricalOptions(3,:) -> double
    % metricalOptions(4,:) -> half-time start first beat
    % metricalOptions(5,:) -> half-time start second beat

end


% run the evaluation code
[cmlC,cmlT,amlC,amlT] = continuity(detections,annotations,tempo_tolerance,phase_tolerance,startUpTime,metricalOptions);

% use amlT as the overall score
mainscore = amlT;
backupscores = [amlC, cmlT, cmlC]; % in case of an amlT tie, we can use these as tie-breakers in this order.

function [cmlC,cmlT,amlC,amlT] = continuity(detections,annotations,tempo_tolerance,phase_tolerance,startUpTime,metricalOptions)

% put the beats and annotations into column vectors
annotations = annotations(:);
detections = detections(:);

% in order to cope with issue where detections or annotations either side of default startUpTime might be excluded 
% we update and specify an individual minAnnotationTime and startUpTime
[minAnnotationTime,minBeatTime] = verifyStartUpTime(startUpTime,annotations,phase_tolerance);

% remove beats and annotations that are before minBeatTime and minAnnotationTime respectively
detections(detections<minBeatTime) = [];
annotations(annotations<minAnnotationTime) = [];

% now do some checks
if (and(isempty(detections),isempty(annotations)))
    cmlC = 1;
    cmlT = 1;
    amlC = 1;
    amlT = 1;
    return
end
if (or(isempty(detections),isempty(annotations)))
    cmlC = 0;
    cmlT = 0;
    amlC = 0;
    amlT = 0;
    return
end

if (length(annotations)<2)
    cmlC = [];
    cmlT = [];
    amlC = [];
    amlT = [];
    disp('At least two annotations (after the minBeatTime) are needed for continuity scores');
    return
end

if (length(detections)<2)
    cmlC = [];
    cmlT = [];
    amlC = [];
    amlT = [];
    disp('At least two detections (after the minBeatTime) are needed for continuity scores');
    return
end

if (or(tempo_tolerance<0,phase_tolerance<0))
    cmlC = [];
    cmlT = [];
    amlC = [];
    amlT = [];
    disp('Tempo and Phase tolerances must be greater than 0');
    return
end


numVariations = size(metricalOptions,1);
variations{numVariations} = [];

for k=1:numVariations,
    startAnn = metricalOptions(k,1);
    factor = metricalOptions(k,2);
    offbeat = metricalOptions(k,3);
    variations{k}=makeVariations(annotations,startAnn,factor,offbeat);
end

% pre-allocate array to store intermediate scores of different variations
cmlCVec = zeros(1,numVariations);
cmlTVec = zeros(1,numVariations);

% loop analysis over number of variants on annotations
for j=1:numVariations,
    [cmlCVec(j),cmlTVec(j)] = ContinuityEval(detections,variations{j},tempo_tolerance,phase_tolerance);
end


% assign the accuracy scores
cmlC = cmlCVec(1);
cmlT = cmlTVec(1);
amlC = max(cmlCVec);
amlT = max(cmlTVec);

function [contAcc, totAcc] = ContinuityEval(detections,annotations,tempo_tolerance,phase_tolerance)
% sub-function for calculating continuity-based accuracy

if (length(annotations)<2)
    contAcc = 0;
    totAcc = 0;
    disp('At least two annotations are required to create an interval');
    return
end

if (length(detections)<2)
    contAcc = 0;
    totAcc = 0;
    disp('At least two detections are required to create an interval');
    return
end

% phase condition
correct_phase = zeros(1,max(length(annotations),length(detections)));
% tempo condition
correct_tempo = zeros(1,max(length(annotations),length(detections)));

for i=1:length(detections)
    
    % find the closest annotation and the signed offset
    [~,closest] = min(abs(annotations-detections(i)));
    signed_offset = detections(i)-annotations(closest);

    % first deal with the phase condition
    tolerance_window = zeros(1,2); % clear each time.
    if (closest==1) % first annotation, so use the forward interval
        annotation_interval = annotations(closest+1)-annotations(closest);
        tolerance_window(1) = -phase_tolerance*(annotation_interval);
        tolerance_window(2) = phase_tolerance*(annotation_interval);
    else % use backward interval
        annotation_interval = annotations(closest)-annotations(closest-1);
        tolerance_window(1) = -phase_tolerance*(annotation_interval);
        tolerance_window(2) = phase_tolerance*(annotation_interval);
    end
    
    % if the signed_offset is within the tolerance window range, then
    % the phase is ok.
    my_eps = 1e-12; % need this to fix rounding errors 
    correct_phase(i) = (signed_offset>=(tolerance_window(1) - my_eps)) && (signed_offset<=(tolerance_window(2) + my_eps));
    
    % now look at the tempo condition
    % calculate the detection interval back to the previous detection
    % (if we can)
    if (i==1) % first detection, so use the interval ahead
        detection_interval = detections(i+1)-detections(i);
    else % we can always look backwards, which is where we should look for the period interval
        detection_interval = detections(i)-detections(i-1);
    end
    
    % find out if the relative intervals of detections to annotations are less than the tolerance
    correct_tempo(i) = ((abs(1-(detection_interval/annotation_interval))) <= (tempo_tolerance));
    
end

% now want to take the logical AND between correct_phase and correct_tempo
correct_beats = correct_phase & correct_tempo;

% we'll look for the longest continuously correct segment
% to do so, we'll add zeros on the front and end in case the sequence is
% all ones
correct_beats = [0 correct_beats(:)' 0];
% now find the boundaries
[~,d2,~] = find(correct_beats==0);
correct_beats = correct_beats(2:end-1);

% in best case, d2 = 1 & length(checkbeats)
contAcc = (max(diff(d2))-1)/length(correct_beats);
totAcc = sum(correct_beats)/length(correct_beats);

function variations = makeVariations(annotations,startAnn,factor,offbeat)

%cut all annotations before first one to keep
annotations = annotations(startAnn:end);

% now interpolate according to factor (factor>1 implies interpolation,
% factor<1 implies sub-sampling)
interpolatedAnnotations = interp1(1:length(annotations),annotations,1:1/factor:length(annotations),'linear');

% if we need make an off-beat version, we interpolate by a factor of two
% again and then take every second annotation
if offbeat==1,
    doubleAnnotations = interp1(1:length(interpolatedAnnotations),interpolatedAnnotations,1:0.5:length(interpolatedAnnotations),'linear');
    variations = doubleAnnotations(2:2:end);
else
    variations = interpolatedAnnotations;
end


function [minAnnotationTime,minBeatTime] = verifyStartUpTime(startUpTime,annotations,phase_tolerance)
% this function aims to cope with the situation where
% an annotation is close to startUpTime (either just ahead or before) 
% and the respective tolerance window would go beyond the startUpTime
% in this situation we now keep the pre-startUpTime annotation
% and specify an individual minAnnotationTime and minBeatTime

% consider three cases:
% 1. annotation just after startUpTime - allow minBeatTime to reflect earliest part of tolerance window (i.e. before startUpTime)
% and set minAnnotationTime equal to the first annotation **after** startUpTime (in effect this is no different to keeping startUpTime) 
% 2. annotation just before startUpTime - again allow minBeatTime to reflect earliest part of tolerance window (i.e. before startUpTime)
% and set minAnnotationTime equal to the last annotation **before** startUpTime
% in fact cases 1. and 2. can be handled identically
% 3. first annotation after startUpTime is sufficiently far ahead that no action is required, 
% i.e. minBeatTime = minAnnotationTime = startUpTime

% find closest annotation to startUpTime
[~,closest] = min(abs(annotations-startUpTime));

% find annotation interval (looking forward) and make backward and forward tolerance windows 
annotation_interval = annotations(closest+1)-annotations(closest);

tolerance_window(1) = -phase_tolerance*annotation_interval;
tolerance_window(2) = phase_tolerance*annotation_interval;

% now check if these tolerance windows straddle startUpTime
if ( ((annotations(closest) + (tolerance_window(1)) < startUpTime)) && ((annotations(closest) + tolerance_window(2)) >= startUpTime) )
	minAnnotationTime = annotations(closest);
	minBeatTime = annotations(closest) + (tolerance_window(1));

else % we don't need to do anything
	minAnnotationTime = startUpTime;
	minBeatTime = startUpTime;
end

% finally double check that minBeatTime and minAnnotationTime can't go below 0
minAnnotationTime = max(0,minAnnotationTime);
minBeatTime = max(0,minBeatTime);
