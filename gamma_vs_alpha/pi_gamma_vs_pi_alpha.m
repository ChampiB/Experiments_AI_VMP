% Fix value of expected free energy
G = [9; 1];

% Values of beta and c
betas  = linspace(0,10);
consts = linspace(10,110);
consts_exp = linspace(10,20);
consts_lin = linspace(10,100);
lin_coef = 10;

% Number of samples per slices
nb_samples = 50;

draw_pi_gamma(G, betas, nb_samples);
draw_pi_alpha(G, consts, nb_samples);
draw_pi_alpha_exp(G, consts_exp, nb_samples);
draw_pi_alpha_lin(G, consts_lin, nb_samples, lin_coef);

function draw_pi_gamma(G, betas, nb_samples)
    [~, columns] = size(betas);
    x = zeros(nb_samples * columns,1);
    y = zeros(nb_samples * columns,1);
    for i = 1:columns
        % gammas = [g_1 g_2 g_3]
        gammas = gamrnd(1,betas(i),nb_samples,1);
        %                      | G_1 * g_1  G_1 * g_2  G_1 * g_3 | (x coord)
        % K = kron(gammas,G) = | G_2 * g_1  G_2 * g_2  G_2 * g_3 | (y coord)
        %                      | G_3 * g_1  G_3 * g_2  G_3 * g_3 | (z coord)
        %
        % SK = softmax(-K) normalises the columns of -K
        %
        % Each column of SK is on point on rectangle at slice position beta
        SK = softmax(-kron(G,gammas'));
        min = (i - 1) * nb_samples + 1;
        max =  i      * nb_samples;
        x(min:max) = ones(nb_samples,1) * betas(i);
        y(min:max) = SK(1,:);
    end
    figure('Name','P(pi|gamma)');
    scatter(x,y);
end

function theta = drchrnd(alpha,n)
    p = length(alpha);
    if size(alpha,2)>size(alpha,1)
        alpha = alpha';
    end
	theta = gamrnd(repmat(alpha,1,n),1,p,n);   
	theta = theta ./ repmat(sum(theta,1),p,1);
end

function draw_pi_alpha(G, consts, nb_samples)
    [~, columns] = size(consts);
    x = zeros(nb_samples * columns,1);
    y = zeros(nb_samples * columns,1);
    for i = 1:columns
        % theta = c - G
        theta = ones(2,1) * consts(i) - G;
        % Sample 'nb_samples' categorical distributions
        alphas = drchrnd(theta, nb_samples);
        % Fill 'x' and 'y'
        min = (i - 1) * nb_samples + 1;
        max =  i      * nb_samples;
        x(min:max) = ones(nb_samples,1) * consts(i);
        y(min:max) = alphas(1,:);
    end
    figure('Name','P(pi|alpha)');
    scatter(x,y);
end

function draw_pi_alpha_exp(G, consts, nb_samples)
    [~, columns] = size(consts);
    x = zeros(nb_samples * columns,1);
    y = zeros(nb_samples * columns,1);
    for i = 1:columns
        % theta = exp(c - G)
        theta = exp(ones(2,1) * consts(i) - G);
        % Sample 'nb_samples' categorical distributions
        alphas = drchrnd(theta, nb_samples);
        % Fill 'x' and 'y'
        min = (i - 1) * nb_samples + 1;
        max =  i      * nb_samples;
        x(min:max) = ones(nb_samples,1) * consts(i);
        y(min:max) = alphas(1,:);
    end
    figure('Name','P(pi|alpha) EXP scale');
    scatter(x,y);
end

function draw_pi_alpha_lin(G, consts, nb_samples, lin_coef)
    [~, columns] = size(consts);
    x = zeros(nb_samples * columns,1);
    y = zeros(nb_samples * columns,1);
    for i = 1:columns
        % theta = lin_coef * (c - G)
        theta = lin_coef * (ones(2,1) * consts(i) - G);
        % Sample 'nb_samples' categorical distributions
        alphas = drchrnd(theta, nb_samples);
        % Fill 'x' and 'y'
        min = (i - 1) * nb_samples + 1;
        max =  i      * nb_samples;
        x(min:max) = ones(nb_samples,1) * consts(i);
        y(min:max) = alphas(1,:);
    end
    figure('Name','P(pi|alpha) Linear coefficient');
    scatter(x,y);
end
