import { Route, Routes } from "react-router";
import { Shell, type PageMeta } from "./components/Shell";
import { Overview } from "./pages/Overview";
import { Running } from "./pages/Running";
import { Pipeline } from "./pages/Pipeline";
import { CaseList, CasePage } from "./pages/Case";
import { ResultPage } from "./pages/Result";
import { FamilyList, FamilyPage } from "./pages/Family";
import { Coverage } from "./pages/Coverage";
import { Inbox } from "./pages/Inbox";
import { TaskPage } from "./pages/Task";
import { Operations } from "./pages/Operations";
import { Agents } from "./pages/Agents";
import { Pool } from "./pages/Pool";
import { Model } from "./pages/Model";
import { Experiments, Improvement } from "./pages/Studies";

const PAGES: PageMeta[] = [
  { path: "/", title: "Overview", question: "What does the system do, and what is it doing now?" },
  { path: "/running", title: "Running", question: "What is occupying the system right now?" },
  { path: "/pipeline", title: "Pipeline", question: "In what order and in parallel did things run, and where did the time go?" },
  { path: "/cases", title: "Cases", question: "Which problems are being solved, and how well?" },
  { path: "/cases/:id", title: "Case", question: "How well is this problem solved, and how did it get here?" },
  { path: "/results/:id", title: "Result", question: "Why is this result trustworthy?" },
  { path: "/families", title: "Families", question: "Which executable families does the pool hold?" },
  { path: "/families/:id", title: "Family", question: "How is this kernel built, step by step?" },
  { path: "/coverage", title: "Coverage", question: "Have we discovered enough, and where are the blind spots?" },
  { path: "/inbox", title: "Inbox", question: "What needs me now?" },
  { path: "/tasks/:id", title: "Task", question: "What did this agent do?" },
  { path: "/operations", title: "Operations", question: "What can I start, stop or change, and what have I done?" },
  { path: "/pool", title: "Pool", question: "What does the system know now?" },
  { path: "/model", title: "Model", question: "Can the performance model be trusted?" },
  { path: "/experiments", title: "Experiments", question: "How far are the experiments, and what do they conclude?" },
  { path: "/improvement", title: "Improvement", question: "Is the system getting better, and where does it fall short?" },
  { path: "/agents", title: "Agents", question: "What controls the worker agents: skills, profile versions, batch environments?" },
];

export function App() {
  return (
    <Shell pages={PAGES}>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/running" element={<Running />} />
        <Route path="/pipeline" element={<Pipeline />} />
        <Route path="/cases" element={<CaseList />} />
        <Route path="/cases/:id" element={<CasePage />} />
        <Route path="/results/:id" element={<ResultPage />} />
        <Route path="/families" element={<FamilyList />} />
        <Route path="/families/:id" element={<FamilyPage />} />
        <Route path="/coverage" element={<Coverage />} />
        <Route path="/inbox" element={<Inbox />} />
        <Route path="/tasks/:id" element={<TaskPage />} />
        <Route path="/operations" element={<Operations />} />
        <Route path="/pool" element={<Pool />} />
        <Route path="/model" element={<Model />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/improvement" element={<Improvement />} />
        <Route path="/agents" element={<Agents />} />
        <Route path="*" element={<div className="empty">No such page.</div>} />
      </Routes>
    </Shell>
  );
}
