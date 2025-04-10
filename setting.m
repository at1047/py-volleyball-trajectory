function setting(xf,yf,v0,theta)

nhat = [cosd(theta); sind(theta)]
v_guess = nhat .* v0

pos_arr = plotting(1,0.01,v_guess(1),v_guess(2));

i = 1;

while pos_arr(1,i) < xf
i = i + 1;
end

plot(pos_arr(1, 1:end),pos_arr(2, 1:end))
axis equal

disp(i)

end
