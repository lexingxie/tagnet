function out_flag = sample_pos_neg(label, neg_pos_ratio, max_num_pos, max_num_neg)

%n = length(label);
ipos = find(label==1) ;
jneg = find(label~=1) ; % 0 or -1

if length(ipos)>max_num_pos
    tmp = randperm(length(ipos));
    ipos = ipos(tmp(1:max_num_pos));
    %fprintf(1, 'capped positive examples at %d\n', max_num_pos);
end

neg_num_cap = min([neg_pos_ratio*length(ipos), max_num_neg]);
if length(jneg)>neg_num_cap
    tmp = randperm(length(jneg));
    jneg = jneg( tmp(1:neg_num_cap) );
    %fprintf(1, 'capped negative examples at %d\n', neg_num_cap);
end


out_flag = false(size(label));
out_flag(ipos) = true;
out_flag(jneg) = true;