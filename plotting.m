function pos_arr = plotting(tmax,dt,v0,theta)

Cd = 0.48;
rho = 1.225;
A = 0.0346;
m = 0.27;

C = Cd * rho * A / (2 * m);

nhat = [cosd(theta); sind(theta)];
v0 = nhat .* v0;

% v0 = [v0x; v0y];
% v0 = [3; 1]
t = 0;

nhat = v0 ./ norm(v0);
Di = C .* v0.^2 .* nhat;

v = [v0(1); Di(1); v0(2); Di(2);];

vel_arr = v0;
pos_arr = [0; 0];
new_pos = pos_arr + v0 .* dt;
pos_arr = [pos_arr, new_pos];


while t < tmax

    dvdt = volleyball(v)

    v = v + dvdt .* dt
    new_vel = [v(1);v(3)];
    vel_arr = [vel_arr, new_vel];

    int_pos = pos_arr(1:2, end);
    new_pos = int_pos + new_vel .* dt;
    pos_arr = [pos_arr, new_pos];

    t = t + dt;
end

figure(1)
plot(pos_arr(1, 1:end),pos_arr(2, 1:end));
axis equal

figure(2)
plot(vel_arr)
axis equal


end
