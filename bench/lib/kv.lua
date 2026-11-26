-- Parse the `key=value` lines the example binaries print to stderr.
-- Numeric values become numbers, anything else stays a string.
local M = {}

function M.parse(text)
    local out = {}
    for line in string.gmatch(text, "[^\n]+") do
        local k, v = string.match(line, "^([%w_]+)=(.*)$")
        if k then
            local n = tonumber(v)
            if n then out[k] = n else out[k] = v end
        end
    end
    return out
end

return M
