-- revised from https://github.com/jarmstrong2/handwritingnet/blob/master/testNet.lua

require 'nn'
require 'torch'
torch.setdefaulttensortype('torch.DoubleTensor')
require 'nngraph'
require 'yHatMat_nocuda'
require 'distributions'
local LSTMH1 = require 'LSTMH1'
local LSTMHN = require 'LSTMHN'
require 'getBatch'
matio = require 'matio'
require 'cunn'

-- load model
model = torch.load('./model/model_full.t7')

-- nearest neighbor retrieved feature to initialize the first axes of the first primitive
test_samp = matio.load('./mydata/train_test.mat')

-- nSamples = 100
nSamples = 900
-- nSamples = 20
maxNumCuboids = 14
-- maxNumCuboids = 6
rs_res = torch.zeros(4*nSamples, 15*maxNumCuboids)

for samp_num = 1, nSamples do
    print(samp_num)
    
x_val = {[1]=test_samp.test_sample[{{1},{samp_num}}][1][1]}
y_val = {[1]=test_samp.test_sample[{{2},{samp_num}}][1][1]}
e_val = {[1]=0}
r_val = {[1]=test_samp.test_sample[{{3},{samp_num}}][1][1]}
s_val = {[1]=test_samp.test_sample[{{4},{samp_num}}][1][1]}    


lstm_c_h1 = torch.zeros(1, 400)
lstm_h_h1 = torch.zeros(1, 400)
lstm_c_h2 = torch.zeros(1, 400)
lstm_h_h2 = torch.zeros(1, 400)
lstm_c_h3 = torch.zeros(1, 400)
lstm_h_h3 = torch.zeros(1, 400)


function makecov(std, rho)
    covmat = torch.Tensor(2,2)
    covmat[{{1},{1}}] = torch.pow(std[{{1},{1}}], 2)
    covmat[{{1},{2}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{1}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{2}}] = torch.pow(std[{{1},{2}}], 2)
    return covmat
end


function getX(input)
    
    e_t = input[{{},{1}}]
    pi_t = input[{{},{2,21}}]
    mu_1_t = input[{{},{22,41}}]
    mu_2_t = input[{{},{42,61}}]
    sigma_1_t = input[{{},{62,81}}]
    sigma_2_t = input[{{},{82,101}}]
    rho_t = input[{{},{102,121}}]
    
    x_3 = torch.Tensor(1)
    x_3 = (x_3:bernoulli(e_t:squeeze())):squeeze()
    --x_3 = e_t:squeeze()

    choice = {}
    
    for k=1,10 do
       table.insert(choice, distributions.cat.rnd(pi_t:squeeze(1)):squeeze()) 
    end
    --chosen_pi = torch.multinomial(pi_t, 1):squeeze()
    --print(chosen_pi)
    _,chosen_pi = torch.max(pi_t,2)
    chosen_pi = chosen_pi:squeeze()
    --print(chosen_pi)

    randChoice = torch.random(10)
    
    max = 0
    for i=1,20 do
        cur = pi_t[{{},{i}}]:squeeze()
        if cur > max then
           max = cur 
            index = i
        end
    end
    
    curstd = torch.Tensor({{sigma_1_t[{{},{chosen_pi}}]:squeeze(), sigma_2_t[{{},{chosen_pi}}]:squeeze()}})
    curcor = torch.Tensor({{1, rho_t[{{},{chosen_pi}}]:squeeze()}})
    curcovmat = makecov(curstd, curcor)
    curmean = torch.Tensor({{mu_1_t[{{},{chosen_pi}}]:squeeze(), mu_2_t[{{},{chosen_pi}}]:squeeze()}})
    sample = distributions.mvn.rnd(curmean, curcovmat)
    x_1 = sample[1]
    x_2 = sample[2]
    --print(e_t)
    --x_1 = curmean[1][1]
    --x_2 = curmean[1][2]

    table.insert(x_val, x_1)
    table.insert(y_val, x_2)
    table.insert(e_val, x_3)
end

--priming the network

for t=1,3*maxNumCuboids do
    x_in = torch.Tensor({{x_val[t], y_val[t], e_val[t]}})
    r_in = torch.Tensor({{r_val[t],s_val[t]}})
    --cond_context = voxMat
-- model 
        
        output_y, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3, rot_res
    = unpack(model:forward({x_in, r_in, lstm_c_h1, lstm_h_h1,
    lstm_c_h2, lstm_h_h2, lstm_c_h3, lstm_h_h3})) 
        
    getX(output_y)
    table.insert(r_val, rot_res[1][1])
    table.insert(s_val, rot_res[1][2])
end

mean_x = test_samp.mean_x[1][1]
mean_y = test_samp.mean_y[1][1]
mean_r = test_samp.mean_r[1][1]
std_x = test_samp.std_x[1][1]
std_y = test_samp.std_y[1][1]
std_r = test_samp.std_r[1][1]

for t=1,3*maxNumCuboids do
    x_val[t] = (x_val[t]*std_x) + mean_x
    y_val[t] = (y_val[t]*std_y) + mean_y
    r_val[t] = (r_val[t]*std_r) + mean_r
end

rs = {}
new = true
count = 0
i = 0
oldx = 0
oldy = 0
for j=1,3*maxNumCuboids do
    i = i + 1
    if new then
        count = count + 1
        table.insert(rs, torch.zeros(4, 3*maxNumCuboids))
        i = 1
        new = false
    end
    
    if e_val[j] == 1 or j == 3*maxNumCuboids then
    --if j == 40 then
        new = true
        rs[count] = rs[count][{{},{1,i}}]
    end
    
    newx = x_val[j]
    newy = y_val[j]
    newr = r_val[j]
    --newx = oldx + x_val[j]
    --newy = oldy - y_val[j] 
    --newy = oldy + y_val[j]
    --oldx = newx
    --oldy = newy
    rs[count][{{1},{i}}] = newx
    rs[count][{{2},{i}}] = newy
    rs[count][{{3},{i}}] = newr
    rs[count][{{4},{i}}] = s_val[j]
    --rs[count][{{5},{i}}] = e_val[j]
end

  
rs_res[{{(samp_num-1)*4+1, samp_num*4},{1,rs[1]:size(2)}}] = rs[1]
    if samp_num % 50 == 0 then
        matio.save('./result/test_res_mn_pure.mat' , rs_res)
    end

end

matio.save('./result/test_res_mn_pure.mat' , rs_res)


--------------------------------------------------------------------------
-- Now, we take this data and convert it back into our format (i.e. JSONs)
--------------------------------------------------------------------------

local json = require 'json'
local lfs = require 'lfs'

lfs.mkdir('./myresult')

local function euler2rotmat(eulerAngs)
    local rx = eulerAngs[1]
    local ry = eulerAngs[2]
    local rz = eulerAngs[3]
    local xmat = torch.Tensor({
        {1, 0, 0},
        {0, math.cos(rx), -math.sin(rx)},
        {0, math.sin(rx), math.cos(rx)},
    })
    local ymat = torch.Tensor({
        {math.cos(ry), 0, math.sin(ry)},
        {0, 1, 0},
        {-math.sin(ry), 0, math.cos(ry)},
    })
    local zmat = torch.Tensor({
        {math.cos(rz), -math.sin(rz), 0},
        {math.sin(rz), math.cos(rz), 0},
        {0, 0, 1},
    })
    return torch.mm(zmat, torch.mm(ymat, xmat))
end

for i=1,nSamples do
    local baseRowIdx = 4*(i-1) + 1
    local cuboids = {}
    local j = 1

    -- Group properties by cuboid and name them
    -- Scan across the group of 4 rows corresponding to this object until
    --    we start seeing zeros in the z-coord position
    while rs_res[baseRowIdx][j+2] ~= 0 do
        local cuboid = {}
        -- Scales
        cuboid.xd = rs_res[baseRowIdx][j]
        cuboid.yd = rs_res[baseRowIdx][j+1]
        cuboid.zd = rs_res[baseRowIdx][j+2]
        -- Translations
        cuboid.origin = torch.Tensor({
            rs_res[baseRowIdx+1][j],
            rs_res[baseRowIdx+1][j+1],
            rs_res[baseRowIdx+1][j+2]
        })
        -- Rotations
        cuboid.eulerAngs = torch.Tensor({
            rs_res[baseRowIdx+2][j],
            rs_res[baseRowIdx+2][j+1],
            rs_res[baseRowIdx+2][j+2]
        })
        table.insert(cuboids, cuboid)
        j = j + 3
    end

    -- Convert Euler angles to axes
    for _,cuboid in ipairs(cuboids) do
        local rotmat = euler2rotmat(cuboid.eulerAngs)
        cuboid.xdir = rotmat[{ {}, 1 }]
        cuboid.ydir = rotmat[{ {}, 2 }]
        cuboid.zdir = rotmat[{ {}, 3 }]
        cuboid.eulerAngs = nil      -- Get rid of this excess property
    end

    -- Convert origins back to centroids
    for _,cuboid in ipairs(cuboids) do
        local c = cuboid.origin
        c = c + 0.5*cuboid.xd*cuboid.xdir
        c = c + 0.5*cuboid.yd*cuboid.ydir
        c = c + 0.5*cuboid.zd*cuboid.zdir
        cuboid.center = c
    end

    -- Add the symmetric duplicates back in (and correctly set their params)
    local all_cuboids = {}
    for _,cuboid in ipairs(cuboids) do
        table.insert(all_cuboids, cuboid)
        -- Cuboid should be duplicated if it is entirely on the negative side of the x=0 plane
        local corners = {
            cuboid.origin,
            cuboid.origin + cuboid.xd*cuboid.xdir,
            cuboid.origin + cuboid.yd*cuboid.ydir,
            cuboid.origin + cuboid.zd*cuboid.zdir,
            cuboid.origin + cuboid.xd*cuboid.xdir + cuboid.yd*cuboid.ydir,
            cuboid.origin + cuboid.xd*cuboid.xdir + cuboid.zd*cuboid.zdir,
            cuboid.origin + cuboid.yd*cuboid.ydir + cuboid.zd*cuboid.zdir,
            cuboid.origin + cuboid.xd*cuboid.xdir + cuboid.yd*cuboid.ydir + cuboid.zd*cuboid.zdir
        }
        cuboid.origin = nil     -- Get rid of this excess property
        local allneg = true
        for _,c in ipairs(corners) do
            allneg = allneg and c[1] <= 0
        end
        if allneg then
            local c = cuboid
            local reflcuboid = {
                xd = c.xd, yd = c.yd, zd = c.zd,
                -- Reflect the centroid about x=0
                center = torch.Tensor({-c.center[1], c.center[2], c.center[3]}),
                -- Flip the x-components of all the direction vectors
                xdir = torch.Tensor({-c.xdir[1], c.xdir[2], c.xdir[3]}),
                ydir = torch.Tensor({-c.ydir[1], c.ydir[2], c.ydir[3]}),
                zdir = torch.Tensor({-c.zdir[1], c.zdir[2], c.zdir[3]})
            }
            table.insert(all_cuboids, reflcuboid)
        end
    end
    cuboids = all_cuboids

    -- Convert list of cuboids into key,value table of cuboids
    local name2cuboid = {}
    for j,cuboid in ipairs(cuboids) do
        -- Also make sure that all the torch Tensors are converted back to Lua tables again
        cuboid.center = torch.totable(cuboid.center)
        cuboid.xdir = torch.totable(cuboid.xdir)
        cuboid.ydir = torch.totable(cuboid.ydir)
        cuboid.zdir = torch.totable(cuboid.zdir)
        name2cuboid['part_' .. j] = cuboid
    end

    -- Save file
    local jsonfile = io.open('./myresult/sample_' .. i .. '.json', "w")
    local jsonstring = json.encode(name2cuboid)
    jsonfile:write(jsonstring)
    jsonfile:close()
end