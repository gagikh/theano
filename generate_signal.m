% Factor to generate points
K = 128;

% Total number of points
M = 16 * K;

% x to approximate
x = 2 * pi * (-M:(M - 1)) / M;

% rand power
P = 0.5;

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

y = sinx_sin2x(x, 0);
dump(x, y, 'signal.txt')

plot(x, y);
pause

S = 16
x = 2 * pi * (-S*M:(S*M - 1)) / M;
y = sinx_sin2x(x, P);
dump(x, y, 'test_signal.txt')

dump(x, y, 'expected_signal.txt')
y = sinx_sin2x(x, 0);
