function dvdt = volleyball(v)

Cd = 0.48;
rho = 1.225;
A = 0.0346;
m = 0.27;

C = Cd * rho * A / (2 * m);

x1 = v(1);
x2 = v(2);
y1 = v(3);
y2 = v(4);

dvdt = zeros(4,1);

dvdt(1) = x2;

dvdt(2) = C * x1^2;

dvdt(3) = y2;

dvdt(4) = C * y1^2 - 9.81;

end
