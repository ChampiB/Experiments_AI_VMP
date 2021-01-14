function MDP = maze_navigation
% Demo of mixed continuous and discrete state space modelling
%__________________________________________________________________________
%
% This demonstration of active inference focuses on navigation and planning
% in a fairly complicated maze. The idea is to demonstrate how epistemic
% foraging and goal (target) directed behaviour are integrated in the
% minimisation of expected free energy. In this illustration, and 8 x 8
% maze is learned through novelty driven evidence accumulation - to learn
% the likelihood mapping between hidden states (locations in the maze) and
% outcomes (whether the current location is open or closed). This
% accumulated experience is then used to plan a path from a start to an end
% (target location) under a task set specified by prior preferences over
% locations. These priors are based upon a simple diffusion (CF backwards
% induction) heuristic that specifies subgoals. The subgoals (i.e.,
% locations) contain the most paths from the target within the horizon of
% the current policy.
%
% We will first illustrate the novelty driven epistemic foraging that
% efficiently scans the maze to learn its structure. We then simulate
% planning of (shortest path) trajectory to the target under the assumption
% the maze has been previously learned. Finally, we consider exploration
% under prior preferences to simulate behaviour when both epistemic and
% goal directed imperatives are in play. The focus on this demo is on
% behavioural and electrophysiological responses over moves.
%
% A key aspect of this formulation is the  hierarchical decomposition of
% goal directed behaviour into subgoals that are within the horizon of a
% limited policy - here, to moves that correspond to a trial. The prior
% preferences then contextualise each policy or trial to ensure that the
% ultimate goal is achieved.
%
% Empirically, this sort of construction suggests the existence of Path
% cells; namely, cells who report the initial location of any subsequence
% and continue firing until the end of the local path. This is illustrated
% by plotting simulated activity as a function of trajectories during 
% exploration.
%
% see also: spm_MPD_VB_X.m
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: DEMO_MDP_maze.m 7766 2020-01-05 21:37:39Z karl $

% set up and preliminaries: first level
%--------------------------------------------------------------------------
rng('default')

% generative model at the sensory level (DEM): continuous states
%==========================================================================
% the generative model has two outcome modalities; namely, what (open
% versus closed) and where (the current location in a maze). These outcomes
% are generated from a single hidden factor (location), where the structure
% of the maze is encoded in the likelihood of observation mapping (that can
% be learned through experience). Allowable actions  include for moves (up,
% down, left, right) and staying at the current location. These induce five
% transition matrices that play the role of empirical priors. Finally,
% prior preferences are based upon allowable transitions (that are function
% of learned accumulated likelihood), which are used to define attractive
% locations within the horizon of two-move policies. These priors implement
% a task set and are returned by a subfunction below: spm_maze_cost
%--------------------------------------------------------------------------
label.factor     = {'where'};
label.modality   = {'distance'};

MAZE  = [...
    1 1 1 1 1 1 1 1;
    1 0 0 0 0 0 0 1;
    1 0 1 1 1 1 0 1;
    1 0 0 0 0 1 0 1;
    1 0 1 1 0 1 0 1;
    1 0 0 0 0 0 0 1;
    1 1 1 1 1 1 1 1];
EXIT_POS  = [2,7];
END       = sub2ind(size(MAZE),2,7);
START     = sub2ind(size(MAZE),6,2);
STATES    = 22;
ACTIONS   = 5;        % UP = 1, DOWN = 2, LEFT = 3, RIGHT = 4, STAY = 5
OUTCOMES  = 10;
[HEIGHT,WIDTH] = size(MAZE);

% prior beliefs about initial states: D 
%--------------------------------------------------------------------------
D{1}  = ones(STATES,1) * 0.1 / (STATES - 1);
D{1}(17,1) = 0.9;

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
A{1} = ones(OUTCOMES,STATES) * 0.1 / (OUTCOMES - 1);    % distance
i = 1;
for x = 1:WIDTH
    for y = 1:HEIGHT
        if (MAZE(y,x) == 0)
            POS = [x,y];
            obs = mahattan_distance(POS, EXIT_POS);
            A{1}(obs, i) = 0.9;
            i = i + 1;
        end
    end
end

% controlled transitions: B (up, down, left, right, stay)
%--------------------------------------------------------------------------
u    = [1 0; -1 0; 0 -1; 0 1; 0 0];               % allowable actions
B{1} = ones(STATES,STATES,ACTIONS) * 0.1 / (STATES - 1);
for y = 1:HEIGHT
    for x = 1:WIDTH
        
        % allowable transitions from state s to state ss
        %------------------------------------------------------------------
        s     = sub2ind([HEIGHT,WIDTH],y,x);
        for k = 1:ACTIONS
            try
                if (MAZE(y + u(k,1),x + u(k,2)) == 0)
                    ss = sub2ind([HEIGHT,WIDTH],y + u(k,1),x + u(k,2));
                    B{1}(ss,s,k) = 0.9;
                else
                    B{1}(s, s,k) = 0.9;
                end
            catch
                B{1}(s, s,k) = 0.9;
            end
        end
    end
end

% allowable policies (2-7 moves): V
%--------------------------------------------------------------------------
V     = [];
for i1 = 1:ACTIONS
for i2 = 1:ACTIONS
%for i3 = 1:ACTIONS
%for i4 = 1:ACTIONS
%for i5 = 1:ACTIONS
%for i6 = 1:ACTIONS
%for i7 = 1:ACTIONS
	V(:,end + 1) = [i1;i2];
%end
%end
%end
%end
%end
end
end
% Execution time for 2-7 moves:
% 2 --> 0.865816
% 3 --> 5.069928
% 4 --> 44.500457
% 5 --> 298.468461
% 6 --> 2642.404988
% 7 --> NEVER ENDED (CRASHED)

% priors: (negative cost) C:
%--------------------------------------------------------------------------
C = zeros(OUTCOMES,1);
for g = 0:OUTCOMES-1
    C(g+1) = OUTCOMES - g;
end
C = spm_softmax(C);

% basic MDP structure
%--------------------------------------------------------------------------
mdp.V = V;                      % allowable policies
mdp.A = A;                      % observation model or likelihood
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states

mdp.label = label;
mdp       = spm_MDP_check(mdp);


% exploratory sequence (with experience and task set)
%==========================================================================
tic
MDP = spm_maze_search(mdp,20,START,END,128,1);
toc

% show results in terms of path
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 3'); clf
spm_maze_plot(MDP,END)

end



    
function res = mahattan_distance(pos1,pos2)
    % Compute the mahatan distance between the two positions
    res = abs(pos1(1) - pos2(1)) + abs(pos1(2) - pos2(2));
end




function MDP = spm_maze_search(mdp,N,START,END,alpha,beta)
% FORMAT MDP = spm_maze_search(mdp,N,START,END,alpha,beta)
% mdp   - MDP structure
% N     - number of trials (i.e., policies: default 8)
% START - index of intial state (default 1)
% END   - index of target state (default 1)
% alpha - prior concentration parameter for likelihood (default 128)
% beta  - precision of prior preference (default 0)
%the argument is
% MDP   - MDP structure array

% preliminaries
%--------------------------------------------------------------------------
try, N;     catch, N     = 8;   end
try, START; catch, START = 1;   end
try, END;   catch, END   = 1;   end
try, alpha; catch, alpha = 128; end
try, beta;  catch, beta  = 0;   end

% initialise concentration parameters: a (if unspecified)
%--------------------------------------------------------------------------
if ~isfield(mdp,'a')
    mdp.a{1} = ones(size(mdp.A{1}))/8 + mdp.A{1}*alpha;
    mdp.a{2} = mdp.A{2}*128;
end
if ~isfield(mdp,'o')
    mdp.o = [];
end
if ~isfield(mdp,'u')
    mdp.u = [];
end
mdp.s = START;

% Evaluate a sequence of moves - recomputing prior preferences at each move
%==========================================================================
for i = 1:N
    
    % Evaluate preferred states (subgoals) on the basis of current beliefs
    %----------------------------------------------------------------------
    mdp.C{2} = spm_maze_cost(mdp,END)*beta;
    
    % proceed with subsequent trial
    %----------------------------------------------------------------------
    MDP(i)   = spm_MDP_VB_X(mdp);
    mdp      = MDP(i);
    mdp.s    = mdp.s(:,end);
    mdp.D{1} = MDP(i).X{1}(:,end);
    mdp.o    = [];
    mdp.u    = [];
    
end
end


function C = spm_maze_cost(MDP,END)
% Evaluate subgoals using graph Laplacian
%==========================================================================
START = MDP.s(1);
if isfield(MDP,'a')
    Q = MDP.a{1};
else
    Q = MDP.A{1};
end
Q   = Q/diag(sum(Q));
Q   = Q(1,:);                                % open states
P   = diag(Q)*any(MDP.B{1},3);               % allowable transitions
ns  = length(Q);                             % number of states
X   = zeros(ns,1);X(START) = 1;              % initial state
Y   = zeros(ns,1);Y(END)   = 1;              % target state


% Preclude transitions to closed states and evaluate graph Laplacian
%--------------------------------------------------------------------------
P   = P - diag(diag(P));
P   = P - diag(sum(P));
P   = expm(P);

% evaluate (negative) cost as a path integral conjunctions
%--------------------------------------------------------------------------
for t = 1:size(MDP.V,1)
    X = P*X;
end
X     = X > exp(-3);
C     = log(X.*(P*Y) + exp(-32));
end



function spm_maze_plot(MDP,END)
% illustrate  search graphically
%--------------------------------------------------------------------------
A  = spm_vec(MDP(1).A{1}(1,:));
ns = numel(A);
ni = sqrt(ns);
A  = reshape(A,ni,ni);
subplot(2,2,1), imagesc(A), axis image
title('Scanpath','fontsize',16);

% Cycle of the trials
%--------------------------------------------------------------------------
h     = [];
MS    = {};
MC    = {};
for p = 1:numel(MDP)
    
    %  current beliefs and preferences: A likelihood
    %----------------------------------------------------------------------
    if isfield(MDP,'a')
        Q = MDP(p).a{1};
    else
        Q = MDP(p).A{1};
    end
    Q     = Q/diag(sum(Q));
    Q     = Q(1,:);
    a     = reshape(Q(:),ni,ni);
    subplot(2,2,2), imagesc(a), axis image
    title('Likelihood','fontsize',16);
    
    %  current beliefs and preferences: B transitions
    %----------------------------------------------------------------------
    try
        b = diag(Q)*any(MDP(p).B{1},3);
    catch
        b = diag(Q)*any(MDP(p).B{1},3);
    end
    subplot(2,2,4), imagesc(-b), axis image
    title('Allowable transitions','fontsize',16);
    
    %  current beliefs and preferences: C preferences
    %----------------------------------------------------------------------
    C     = MDP(p).C{2}(:,1);
    C     = spm_softmax(C);
    C     = reshape(C,ni,ni);
    subplot(2,2,3), imagesc(C), axis image
    title('Preferences','fontsize',16);
    try
        [i,j] = ind2sub([ni,ni],MDP(p).s(1)); hold on
        plot(j,i,'.','MarkerSize',32,'Color','g');
        [i,j] = ind2sub([ni,ni],END);
        plot(j,i,'.','MarkerSize',32,'Color','r'); hold off
    end
    
    % cycle over  short-term searches
    %----------------------------------------------------------------------
    subplot(2,2,1),hold on
    s     = MDP(p).s;
    for t = 1:numel(s)
        
        % location
        %------------------------------------------------------------------
        [i,j] = ind2sub([ni,ni],s(t));
        h(end + 1) = plot(j,i,'.','MarkerSize',32,'Color','r');
        try
            set(h(end - 1),'Color','m','MarkerSize',16);
            j = [get(h(end - 1),'Xdata'), get(h(end),'Xdata')];
            i = [get(h(end - 1),'Ydata'), get(h(end),'Ydata')];
            plot(j,i,':r');
        end
        
        % save
        %------------------------------------------------------------------
        if numel(MS)
            MS(end + 1) = getframe(gca);
        else
            MS = getframe(gca);
        end
        
    end
    
    % save
    %----------------------------------------------------------------------
    subplot(2,2,3)
    if numel(MC)
        MC(end + 1) = getframe(gca);
    else
        MC = getframe(gca);
    end
    
end

% save movie
%--------------------------------------------------------------------------
subplot(2,2,1)
xlabel('click axis for movie')
set(gca,'Userdata',{MS,16})
set(gca,'ButtonDownFcn','spm_DEM_ButtonDownFcn')

subplot(2,2,3)
xlabel('click axis for movie')
set(gca,'Userdata',{MC,16})
set(gca,'ButtonDownFcn','spm_DEM_ButtonDownFcn')
end

