using Oceananigans.OutputWriters: AbstractSchedule

import Oceananigans: initialize!, prognostic_state, restore_prognostic_state!

mutable struct RepeatingIntervals{FT} <: AbstractSchedule
    times :: Vector{FT}
    previous_actuation :: Int
    loops :: Int
end

function RepeatingIntervals(times...)
    length(times) == 0 && return RepeatingIntervals(Float64[], 0)

    first_time = times[1]

    if all(t -> t isa Number, times)
        FT = Oceananigans.defaults.FloatType
        return RepeatingIntervals{FT}(sort([convert(FT, t) for t in times]), 0, 0)
    elseif all(t -> t isa AbstractTime, times)
        TT = typeof(first_time)
        return RepeatingIntervals{TT}(sort(collect(times)), 0, 0)
    else
        throw(ArgumentError("RepeatingIntervals expects all times to be numbers or all to be Date/DateTime."))
    end
end

function next_actuation_time(st::RepeatingIntervals)
    if st.previous_actuation >= length(st.times)
        st.previous_actuation = 0
        st.loops += 1
    end
    return st.times[st.previous_actuation+1] + st.times[end] * st.loops
end

function (st::RepeatingIntervals)(model)
    current_time = model.clock.time
    next_time = next_actuation_time(st)

    if current_time >= next_time
        st.previous_actuation += 1
        return true
    end

    return false
end

initialize!(st::RepeatingIntervals, model) = st(model)

function schedule_aligned_time_step(schedule::RepeatingIntervals, clock, Δt)
    t★ = next_actuation_time(schedule)
    δt = t★ == Inf ? Δt : time_difference_seconds(t★, clock.time)
    return min(Δt, δt)
end

Base.copy(st::RepeatingIntervals) = RepeatingIntervals(copy(st.times), st.previous_actuation)

function prognostic_state(schedule::RepeatingIntervals)
    return (; previous_actuation = schedule.previous_actuation)
end

function restore_prognostic_state!(restored::RepeatingIntervals, from)
    restored.previous_actuation = from.previous_actuation
    return restored
end

restore_prognostic_state!(::RepeatingIntervals, ::Nothing) = nothing

