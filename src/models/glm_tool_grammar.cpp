#include "models/glm_tool_grammar.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "models/glm_tokenizer.hpp"

namespace dgpp::glm {

// ---------------------------------------------------------------------------
// GrammarVocab
// ---------------------------------------------------------------------------

GrammarVocab::GrammarVocab(std::vector<std::string> texts, ChatMarkers markers,
                           std::vector<int64_t> eos_ids, int vocab_size,
                           int64_t call_turn_eos)
    : texts_(std::move(texts)),
      markers_(std::move(markers)),
      eos_(std::move(eos_ids)),
      call_eos_(call_turn_eos),
      vocab_size_(vocab_size) {
  if (vocab_size_ < 1)
    throw std::invalid_argument("GrammarVocab: vocab_size must be positive");
  for (size_t id = 0; id < texts_.size(); ++id) {
    if (texts_[id].empty() || static_cast<int64_t>(id) >= vocab_size_) continue;
    by_first_[static_cast<unsigned char>(texts_[id][0])].push_back(
        static_cast<int32_t>(id));
  }
  for (const ChatMarker* m :
       {&markers_.tool_call_open, &markers_.tool_call_close,
        &markers_.arg_key_open, &markers_.arg_key_close,
        &markers_.arg_value_open, &markers_.arg_value_close})
    if (m->available()) marker_ids_.push_back(m->id);
  if (call_eos_ < 0 && !eos_.empty()) call_eos_ = eos_[0];
}

GrammarVocab GrammarVocab::from_tokenizer(const GlmTokenizer& tok,
                                          const std::vector<int64_t>& eos_ids,
                                          int vocab_size) {
  const int64_t max_id = tok.max_id();
  std::vector<std::string> texts(
      static_cast<size_t>(std::max<int64_t>(max_id + 1, 0)));
  for (int64_t id = 0; id <= max_id; ++id)
    texts[static_cast<size_t>(id)] = tok.decode(id, /*skip_special_tokens=*/true);
  // The turn-ending id after the last call: <|observation|> when it is an
  // EOS id of this checkpoint, else the first EOS id.
  int64_t call_eos = -1;
  for (const auto& added : tok.added_tokens())
    if (added.content == "<|observation|>")
      for (const int64_t e : eos_ids)
        if (e == added.id) call_eos = e;
  return GrammarVocab(std::move(texts), ChatMarkers::from_tokenizer(tok),
                      eos_ids, vocab_size, call_eos);
}

bool GrammarVocab::is_eos(int64_t id) const {
  for (const int64_t e : eos_)
    if (e == id) return true;
  return false;
}

// ---------------------------------------------------------------------------
// GrammarState
// ---------------------------------------------------------------------------

GrammarState::GrammarState(const GrammarVocab* vocab, GrammarSpec spec,
                           bool prompt_opens_thinking)
    : vocab_(vocab), spec_(std::move(spec)) {
  if (!spec_.active()) return;
  if (vocab_ == nullptr || !vocab_->usable())
    throw std::invalid_argument(
        "GrammarState: an active grammar needs a vocabulary with the "
        "tool-call markers and an EOS id");
  if (spec_.mode == GrammarSpec::Mode::kNamed) {
    bool found = false;
    for (const GrammarTool& t : spec_.tools) found = found || t.name == spec_.named;
    if (!found)
      throw std::invalid_argument(
          "GrammarState: the named function is not among the tools");
  }
  if ((spec_.mode == GrammarSpec::Mode::kRequired ||
       spec_.mode == GrammarSpec::Mode::kNamed) &&
      spec_.tools.empty())
    throw std::invalid_argument("GrammarState: a required call needs tools");
  state_ = prompt_opens_thinking && vocab_->markers().think_close.available()
               ? State::kThink
               : State::kTop;
}

const char* GrammarState::state_name() const {
  if (!active()) return dead_ ? "dead" : "inactive";
  switch (state_) {
    case State::kThink: return "think";
    case State::kTop: return "top";
    case State::kName: return "name";
    case State::kKey: return "key";
    case State::kAfterKey: return "after-key";
    case State::kValue: return "value";
    case State::kAfterValue: return "after-value";
    case State::kEnd: return "end";
    case State::kDone: return "done";
  }
  return "?";
}

bool GrammarState::obligation_open() const {
  return (spec_.mode == GrammarSpec::Mode::kRequired ||
          spec_.mode == GrammarSpec::Mode::kNamed) &&
         calls_ == 0;
}

bool GrammarState::calls_remaining() const {
  switch (spec_.mode) {
    case GrammarSpec::Mode::kNone: return true;
    case GrammarSpec::Mode::kForbidCalls: return false;
    case GrammarSpec::Mode::kAuto: return calls_ == 0;
    case GrammarSpec::Mode::kRequired: return spec_.parallel || calls_ == 0;
    case GrammarSpec::Mode::kNamed: return calls_ == 0;
  }
  return false;
}

const GrammarTool* GrammarState::current_tool() const {
  return tool_ >= 0 && tool_ < static_cast<int>(spec_.tools.size())
             ? &spec_.tools[static_cast<size_t>(tool_)]
             : nullptr;
}

bool GrammarState::keys_possible() const {
  const GrammarTool* t = current_tool();
  if (t == nullptr) return true;
  return !t->constrain_keys || !t->keys.empty();
}

void GrammarState::enter(State s) {
  state_ = s;
  match_ = TextMatch{};
  if (s == State::kName) {
    if (spec_.mode == GrammarSpec::Mode::kNamed) {
      match_.targets.push_back(spec_.named);
    } else {
      for (const GrammarTool& t : spec_.tools) match_.targets.push_back(t.name);
    }
    tool_ = -1;
  } else if (s == State::kKey) {
    const GrammarTool* t = current_tool();
    if (t != nullptr && t->constrain_keys) match_.targets = t->keys;
  } else if (s == State::kTop) {
    tool_ = -1;
  }
}

std::vector<int64_t> GrammarState::match_ids(const TextMatch& m,
                                             int64_t closer) const {
  std::vector<int64_t> out;
  for (const std::string& target : m.targets) {
    if (target.size() <= m.emitted.size()) continue;
    if (target.compare(0, m.emitted.size(), m.emitted) != 0) continue;
    const size_t remaining = target.size() - m.emitted.size();
    const unsigned char b = static_cast<unsigned char>(target[m.emitted.size()]);
    for (const int32_t id : vocab_->ids_starting_with(b)) {
      const std::string& text = vocab_->text(id);
      if (text.size() <= remaining &&
          target.compare(m.emitted.size(), text.size(), text) == 0)
        out.push_back(id);
    }
  }
  if (m.complete() && closer >= 0) out.push_back(closer);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

void GrammarState::free_mask(TokenMask* out, int64_t extra_allowed,
                             bool forbid_markers) const {
  const int vocab = vocab_->vocab_size();
  out->vocab = vocab;
  out->words.assign(static_cast<size_t>(TokenMask::words_for(vocab)), 0xffffffffu);
  int allowed = vocab;
  // The pad bits past the vocabulary are cleared for tidiness (never read).
  if (vocab % 32 != 0)
    out->words.back() &= (1u << (vocab % 32)) - 1u;
  const auto clear = [&](int64_t id) {
    if (id < 0 || id >= vocab || id == extra_allowed) return;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (w & bit) {
      w &= ~bit;
      --allowed;
    }
  };
  if (forbid_markers)
    for (const int64_t m : vocab_->marker_ids()) clear(m);
  if (obligation_open())
    for (const int64_t e : vocab_->eos_ids()) clear(e);
  out->allowed = allowed;
}

void GrammarState::list_mask(TokenMask* out,
                             const std::vector<int64_t>& ids) const {
  const int vocab = vocab_->vocab_size();
  out->vocab = vocab;
  out->words.assign(static_cast<size_t>(TokenMask::words_for(vocab)), 0u);
  int allowed = 0;
  for (const int64_t id : ids) {
    if (id < 0 || id >= vocab) continue;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (!(w & bit)) {
      w |= bit;
      ++allowed;
    }
  }
  out->allowed = allowed;
  if (allowed == 0)
    throw std::logic_error("GrammarState: a position with no allowed id");
}

void GrammarState::mask(TokenMask* out) const {
  out->allowed = 0;
  if (!active()) return;
  const ChatMarkers& m = vocab_->markers();
  switch (state_) {
    case State::kThink:
      // Free (the markers included — reasoning is the model's), but the
      // turn may not end while a call is owed.
      if (!obligation_open()) return;  // unconstrained
      free_mask(out, /*extra_allowed=*/-1, /*forbid_markers=*/false);
      return;
    case State::kTop: {
      const bool free_top = spec_.mode == GrammarSpec::Mode::kAuto ||
                            spec_.mode == GrammarSpec::Mode::kForbidCalls;
      if (free_top) {
        free_mask(out, calls_remaining() ? m.tool_call_open.id : -1);
        return;
      }
      std::vector<int64_t> ids;
      if (calls_remaining()) ids.push_back(m.tool_call_open.id);
      if (!obligation_open()) ids.push_back(vocab_->call_turn_eos());
      list_mask(out, ids);
      return;
    }
    case State::kName: {
      std::vector<int64_t> ids = match_ids(match_, -1);
      if (match_.complete()) {
        if (keys_possible_for(match_.emitted)) ids.push_back(m.arg_key_open.id);
        ids.push_back(m.tool_call_close.id);
      }
      list_mask(out, ids);
      return;
    }
    case State::kKey: {
      const GrammarTool* t = current_tool();
      if (t != nullptr && t->constrain_keys) {
        list_mask(out, match_ids(match_, m.arg_key_close.id));
        return;
      }
      free_mask(out, m.arg_key_close.id);
      return;
    }
    case State::kAfterKey:
      list_mask(out, {m.arg_value_open.id});
      return;
    case State::kValue:
      free_mask(out, m.arg_value_close.id);
      return;
    case State::kAfterValue: {
      std::vector<int64_t> ids;
      if (keys_possible()) ids.push_back(m.arg_key_open.id);
      ids.push_back(m.tool_call_close.id);
      list_mask(out, ids);
      return;
    }
    case State::kEnd:
    case State::kDone:
      list_mask(out, {vocab_->call_turn_eos()});
      return;
  }
}

bool GrammarState::keys_possible_for(const std::string& name) const {
  for (const GrammarTool& t : spec_.tools)
    if (t.name == name) return !t.constrain_keys || !t.keys.empty();
  return true;
}

bool GrammarState::allows(int64_t id) const {
  if (!active()) return true;
  TokenMask m;
  mask(&m);
  return m.allows(id);
}

void GrammarState::advance(int64_t id) {
  if (!active()) return;
  if (!allows(id)) {
    dead_ = true;
    return;
  }
  const ChatMarkers& m = vocab_->markers();
  const auto after_call = [&] {
    ++calls_;
    if (calls_remaining())
      enter(State::kTop);
    else
      enter(State::kEnd);
  };
  switch (state_) {
    case State::kThink:
      if (id == m.think_close.id) enter(State::kTop);
      return;
    case State::kTop:
      if (id == m.tool_call_open.id) {
        enter(State::kName);
      } else if (vocab_->is_eos(id)) {
        state_ = State::kDone;
      }
      return;
    case State::kName:
      if (id == m.arg_key_open.id || id == m.tool_call_close.id) {
        // The name is complete: bind the tool.
        tool_ = -1;
        for (size_t i = 0; i < spec_.tools.size(); ++i)
          if (spec_.tools[i].name == match_.emitted) tool_ = static_cast<int>(i);
        if (id == m.arg_key_open.id)
          enter(State::kKey);
        else
          after_call();
        return;
      }
      match_.emitted += vocab_->text(id);
      return;
    case State::kKey:
      if (id == m.arg_key_close.id) {
        enter(State::kAfterKey);
        return;
      }
      match_.emitted += vocab_->text(id);
      return;
    case State::kAfterKey:
      enter(State::kValue);
      return;
    case State::kValue:
      if (id == m.arg_value_close.id) enter(State::kAfterValue);
      return;
    case State::kAfterValue:
      if (id == m.arg_key_open.id)
        enter(State::kKey);
      else
        after_call();
      return;
    case State::kEnd:
      state_ = State::kDone;
      return;
    case State::kDone:
      return;
  }
}

}  // namespace dgpp::glm
