function MDP = maze_navigation
rng('default')
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
EXIT_POS    = [2,7];
START_POS   = [6,2];
STATES      = 22;   % Number of states
ACTIONS     = 5;    % Number of actions: UP=1,DOWN=2,LEFT=3,RIGHT=4,STAY=5
OUTCOMES    = 10;   % Number of outcomes
YES         = 1;
NO          = 0;
TRIALS      = 15;

% Load mapping from position to state index
%--------------------------------------------------------------------------
STATES_INDEX = (-1) * ones(size(MAZE));
i = 1;
for y = 1:size(MAZE,1)
    for x = 1:size(MAZE,2)
        if (MAZE(y,x) == 0)
            STATES_INDEX(y,x) = i;
            i = i + 1;
        end
    end
end
EXIT_STATE  = STATES_INDEX(EXIT_POS(1), EXIT_POS(2) );
START_STATE = STATES_INDEX(START_POS(1),START_POS(2));

% prior beliefs about initial states: D 
%--------------------------------------------------------------------------
D{1} = ones(STATES,1) * NO / (STATES - 1);
D{1}(START_STATE,1) = YES;

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
A{1} = ones(OUTCOMES,STATES) * NO / (OUTCOMES - 1);    % distance
for y = 1:size(MAZE,1)
    for x = 1:size(MAZE,2)
        if (MAZE(y,x) == 0)
            POS = [y,x];
            obs = mahattan_distance(POS, EXIT_POS);
            A{1}(obs + 1, STATES_INDEX(y,x)) = YES;
        end
    end
end

% controlled transitions: B (up, down, left, right, stay)
%--------------------------------------------------------------------------
u    = [-1 0; 1 0; 0 -1; 0 1; 0 0];               % allowable actions
B{1} = ones(STATES,STATES,ACTIONS) * NO / (STATES - 1);
for y = 1:size(MAZE,1)
    for x = 1:size(MAZE,2)
        % allowable transitions from state s to state ss
        %------------------------------------------------------------------
        s = STATES_INDEX(y,x);
        if (MAZE(y,x) == 0)
        for k = 1:ACTIONS
            try
                if (MAZE(y + u(k,1),x + u(k,2)) == 0)
                    ss = STATES_INDEX(y + u(k,1),x + u(k,2));
                    B{1}(ss,s,k) = YES;
                else
                    B{1}(s,s,k) = YES;
                end
            catch
                B{1}(s,s,k) = YES;
            end
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

% 2-5 moves - softmax leads to local minimum
% 5   moves + softmax leads to local minimum
% 4   moves + softmax leads to global minimum
% 3   moves + softmax leads to state 19
% 3   moves + softmax leads to state 13

% priors: (negative cost) C:
%--------------------------------------------------------------------------
C{1} = zeros(OUTCOMES,1);
for g = 1:OUTCOMES
    C{1}(g) = OUTCOMES - g - 1;
end
C{1} = spm_softmax(C{1});

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
MDP = spm_maze_search(mdp,TRIALS,START_STATE,EXIT_STATE,128,1);
toc

% show results in terms of path
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1'); clf
spm_maze_plot(MDP,MAZE,STATES_INDEX);
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
end
if ~isfield(mdp,'o')
    mdp.o = [];
end
if ~isfield(mdp,'u')
    mdp.u = [];
end
mdp.s = START;

% Evaluate a sequence of moves
%==========================================================================
for i = 1:N
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

function res = invert_color(MAZE)
res = zeros(size(MAZE));
for y = 1:size(MAZE,1)
    for x = 1:size(MAZE,2)
        if (MAZE(y,x) == 0)
            res(y,x) = 1;
        end
    end
end
end

function [yp,xp] = state_to_position(STATES_INDEX,state)
for y = 1:size(STATES_INDEX,1)
    for x = 1:size(STATES_INDEX,2)
        if (state == STATES_INDEX(y,x))
            xp = x;
            yp = y;
            return
        end
    end
end
end

function spm_maze_plot(MDP,MAZE,STATES_INDEX)
% display maze
%--------------------------------------------------------------------------
MAZE = invert_color(MAZE);
subplot(2,2,1), imagesc(MAZE), axis image
title('Scanpath','fontsize',16);

% Cycle of the trials
%--------------------------------------------------------------------------
h     = [];
MS    = {};
MC    = {};
for p = 1:numel(MDP)
    %  display prior preferences
    %----------------------------------------------------------------------
    C     = MDP(p).C{1}(:,1);
    C     = spm_softmax(C);
    subplot(2,2,3), imagesc(C), axis image
    title('Preferences','fontsize',16);
    
    % cycle over  short-term searches
    %----------------------------------------------------------------------
    subplot(2,2,1),hold on
    s     = MDP(p).s;
    for t = 1:numel(s)
        % location
        %------------------------------------------------------------------
        [i,j] = state_to_position(STATES_INDEX,s(t));
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
