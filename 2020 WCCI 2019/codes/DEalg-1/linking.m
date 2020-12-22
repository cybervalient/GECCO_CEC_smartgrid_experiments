function [ seps,grouped_nonseps] = linking(seps,nonseps,lb,ub)    
    xremain = [];    
    oneday_dim = numel(lb)/24;
    idx = oneday_dim;
    for t = 1:12
        xremain = [xremain,oneday_dim+ oneday_dim*2*(t-1)+1:oneday_dim+oneday_dim*2*(t-1)+oneday_dim];
    end
   
    


    for i = 1:numel(nonseps)
        for j = 1:numel(nonseps{i})
            nonseps{i} = [nonseps{i}, nonseps{i}(j)+idx];
            xremain(find(xremain == (nonseps{i}(j)+idx))) = [];      
        end    
    end
    
    

    grouped_nonseps = [];
    for i = 1:numel(nonseps)
        grouped_nonseps =[grouped_nonseps, nonseps{i}]; 
        
    end
    seps = [seps;xremain']';
    
    
     dummy = find(lb == ub);
     for i = 1:numel(dummy)
         if ismember(dummy(i),seps)
             seps(find(dummy(i)==seps)) = [];
         end 
         if ismember(dummy(i),grouped_nonseps)
             grouped_nonseps(find(dummy(i)==grouped_nonseps)) = [];
         end 

     end

    
    
    

end

