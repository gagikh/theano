% Factor to generate points
K = 128;

% Total number of points
M = 1024 * K;

% x to approximate
x = 2 * pi * (-M:(M - 1)) / M;

% rand power
P = 0.01;

function dump(x, y, name)
    z(1, :) = x;
    z(2, :) = y;
    f = fopen(name, 'w');
    fprintf(f, '%d, %d\n', z);
    fclose(f);
end

function y = sinx_sin2x(x, P)
    y = sin(3 * x) .* sin(x) + P * rand(1, length(x));
end

y = sinx_sin2x(x, P);
plot(x, y);
pause

dump(x, y, 'signal.txt')
