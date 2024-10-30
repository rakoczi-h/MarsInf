function p_all = precompute_Legendre(num_rows)

L = num_rows-1;
dt = double(180/num_rows);
theta = double((dt/2:dt:180))';
p_all = struct();
for m = double(0:L)
    p = Legendre_functions(m:L, m, theta);
    name = strjoin({'m', num2str(m)}, '');
    savename = strjoin({'legendres/legendre_',num2str(L),'_',num2str(m),'.mat'}, '');
    save(savename, 'p');
    p_all.(name) = p;
end

