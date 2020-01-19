local argparse = require "argparse"
local json = require "json"
local lfs = require "lfs"
local matio = require "matio"
local path = require "path"

local parser = argparse("prepareTrainingData", "Convert directories of .JSON shape files into .t7 and .mat files")
parser:argument("train_dir", "Directory containing training data")
parser:argument("val_dir", "Directory containing validation data")
parser:option("-ot --output_dir", "Directory for output files", "data")
local args = parser:parse()

local function cs(n)
    local str = ""
    for i=1,n do
        str = str .. "c"
    end
    return str
end

local function zeros(n)
    local z = {}
    for i=1,n do
        table.insert(z, 0)
    end
    return z
end

local function addall(dst, src)
    for _,x in ipairs(src) do
        table.insert(dst, x)
    end
end

-- https://www.gregslabaugh.net/publications/euler.pdf
local function rotmat2euler(R)
    local theta
    local phi
    local psi
    if math.abs(R[3][1]) ~= 1 then
        -- Two valid solutions
        local theta1 = -math.asin(R[3][1])
        local theta2 = math.pi - theta1
        local psi1 = math.atan2(R[3][2]/math.cos(theta1), R[3][3]/math.cos(theta1))
        local psi2 = math.atan2(R[3][2]/math.cos(theta2), R[3][3]/math.cos(theta2))
        local phi1 = math.atan2(R[2][1]/math.cos(theta1), R[1][1]/math.cos(theta1))
        local phi2 = math.atan2(R[2][1]/math.cos(theta2), R[1][1]/math.cos(theta2))
        -- WLOG, let's pick the first one
        theta = theta1
        phi = phi1
        psi = psi1
    else
        -- phi can be anything; for simplicity, choose 0
        phi = 0
        if R[3][1] == -1 then
            theta = math.pi/2
            psi = phi + math.atan2(R[1][2], R[1][3])
        else
            theta = -math.pi/2
            psi = -phi + math.atan2(-R[1][2], -R[1][3])
        end
    end
    -- Return the x, y, z Euler angles 
    return {psi, theta, phi}
end

local function convertDir(indir, outbasename, stats)

    local all_data = {}
    local all_x = {}
    local all_y = {}
    local all_r = {}

    for entry in lfs.dir(indir) do
        local nameAndExt = path.nameext(entry)
        if #nameAndExt == 2 and nameAndExt[2] == 'json' do
            local jsonfile = io.open(path.combine(dirname, entry), "r")
            local jsonstring = jsonfile:read("*a")
            local jsonfile:close()
            local jsondata = json.decode(jsonstring)

            local cuboids = {}
            for partname, partdata in pairs(jsondata) do
                table.insert(cuboids, partdata)
            end

            -- Sort by decreasing centroid height
            table.sort(cuboids, function (a, b)
                return a.center[2] > b.center[2]
            end)

            -- Convert axes into Euler angles
            -- First construct change-of-basis rotation matrix, then extract
            --    Euler angles from that matrix
            for _,cuboid in cuboids do
                local basismat = torch.zeros(3, 3)
                basismat[{ {}, 1 }] = cuboids.xdir
                basismat[{ {}, 2 }] = cuboids.ydir
                basismat[{ {}, 3 }] = cuboids.zdir
                cuboid.eulerAngs = rotmat2euler(basismat)
            end

            -- Convert to expected training data format
            local data = {
                str = cs(#cuboids),           -- one "c" for each cuboid
                s_vals = zeros(3*#cuboids),   -- symmetries; unused
                rs_vals = zeros(3*#cuboids),  -- rotation axis
                x_vals = zeros(3*#cuboids),   -- scale
                e_vals = zeros(3*#cuboids)    -- stop signal
                y_vals = zeros(3*#cuboids),   -- translation
                r_vals = zeros(3*#cuboids)    -- rotation amount
            }
            data.e_vals[#data._evals] = 1
            
            for i,cuboid in ipairs(cuboids) do
                for j=1,3 do
                    local index = 3*(i-1) + (j-1) + 1
                    -- Translations
                    data.y_vals[index] = cuboid.center[j]
                    -- Rotations
                    if cuboid.eulerAngs[j] ~= 0 then
                        data.rs_vals[index] = 1
                        data.r_vals[index] = cuboid.eulerAngs[j]
                    end
                end
                -- Scale
                local index = 3*(i-1) + 1
                data.x_vals[index] = cuboid.xd
                data.x_vals[index+1] = cuboid.yd
                data.x_vals[index+2] = cuboid.zd
            end

            -- Add to the overall table of data
            table.insert(all_data, data)
            addall(all_x, data.x_vals)
            addall(all_y, data.y_vals)
            addall(all_r, data.r_vals)
        end  
    end

    -- Normalize x_vals, y_vals, and r_vals to be zero mean and unit variance
    if stats == nil then
        local x_tensor = torch.Tensor(all_x)
        local y_tensor = torch.Tensor(all_y)
        local r_tensor = torch.Tensor(all_r)
        local x_mean = torch.mean(x_tensor)
        local y_mean = torch.mean(y_tensor)
        local r_mean = torch.mean(r_tensor)
        local x_std = torch.std(x_tensor)
        local y_std = torch.std(y_tensor)
        local r_std = torch.std(r_tensor)
        stats = {
            x_mean = x_mean,
            y_mean = y_mean, 
            r_mean = r_mean, 
            x_std = x_std, 
            y_std = y_std, 
            r_std = r_std
        }
    end
    for _,data in ipairs(all_data) do
        for i,#data.x_vals do
            data.x_vals[i] = (data.x_vals[i] - stats.x_mean) / stats.x_std
            data.y_vals[i] = (data.y_vals[i] - stats.y_mean) / stats.y_std
            data.r_vals[i] = (data.r_vals[i] - stats.r_mean) / stats.r_std
        end
    end

    -- Save as a torch .t7 file
    local file = torch.DiskFile(outbasename .. '_all.t7')
    file:writeObject(all_data)
    file:close()

    -- Save a .mat file with the initial values needed to prime the network
    local test_ret_num = torch.zeros(#all_data, 1)  -- Values aren't important; only the size is used
    local test_sample = torch.zeros(5, #all_data)
    for i,data in ipairs(all_data) do
        -- Order is x, y, e, r, rs
        test_sample[1, i] = data.x_vals[1]
        test_sample[2, i] = data.y_vals[1]
        test_sample[3, i] = data.e_vals[1]
        test_sample[4, i] = data.r_vals[1]
        test_sample[5, i] = data.rs_vals[1]
    end
    matio.save(outbasename .. '_prime.mat', {test_ret_num = test_ret_num, test_sample = test_sample})

    -- Save a .mat file with *everything*, so we can run it through the Matlab visualization code
    --    and verify that it looks correct
    local maxNumCuboids = 0
    for _,data in ipairs(all_data) do
        maxNumCuboids = math.max(maxNumCuboids, #data.str)
    end
    local all_data_mat = torch.zeros(#all_data, 15*maxNumCuboids)
    for i,data in ipairs(all_data) do
        local n = #data.x_vals
        for j=1,n do
            local index = 5*(j-1) + 1
            all_data_mat[i, index] = data.x_vals[j]
            all_data_mat[i, index+1] = data.y_vals[j]
            all_data_mat[i, index+2] = data.e_vals[j]
            all_data_mat[i, index+3] = data.r_vals[j]
            all_data_mat[i, index+4] = data.rs_vals[j]
        end
    end
    matio.save(outbasename .. '_all.mat', all_data_mat)

    -- Return the means and standard deviations (so they can be saved and later used to convert
    --    data back into legible form)
    return stats
end

-- Convert training data
stats = convertDir(args.train_dir, args.output_dir .. '/train')

-- Convert validation data, normalizing with same statistics as the training data
convertDir(args.val_dir, args.output_dir .. '/val', stats)

-- Save statistics
local file = io.open(args.output_dir .. '/stats.json', 'w')
file:write(json.encode(stats))
file:close()

