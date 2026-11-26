-- Shared script for every example-binary bench. The bench table's `target`
-- picks the example; the example takes no arguments, does its own timing and
-- prints its numbers as key=value lines on stderr, which land in `record`.
local kv = require("kv")

function bench()
    local r = binary_with({ hotpath = "run", measured = true }):ok()
    local values = kv.parse(r.stderr)
    if values.elapsed_ms == nil then
        fail("no elapsed_ms line on stderr")
    end
    record.kv = values
end
