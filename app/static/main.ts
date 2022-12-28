import { Model, SubModel, SubModelSteps, ImageSet, Image } from "./base_types.js"
import { sort, createElement } from "./util.js"
import { StoredVal } from "./storage.js"

const MODEL_FIELDS = ["name", "base"]
const SUBMODEL_FIELDS = ["submodelStr", "seed", "batch", "learningRate"]
const DESELECTED = "deselected"

const STORE_VIS_MODEL = new StoredVal('vis_model', new Set<string>(), (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))
const STORE_VIS_SUBMODEL = new StoredVal('vis_submodel', new Set<string>(), (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))
const STORE_VIS_SUBMODEL_STEPS = new StoredVal('vis_submodelSteps', new Set<string>(), (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))
const STORE_VIS_PROMPT = new StoredVal('vis_prompt', new Set<string>(), (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))
const STORE_VIS_SAMPLER = new StoredVal('vis_sampler', new Set<string>(), (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))
const STORE_VIS_CFG = new StoredVal('vis_cfg', new Set<number>(), (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))
const STORE_VIS_RESOLUTION = new StoredVal('vis_resolution', new Set<string>(), (storage) => Array.from(storage), (jsonVal) => new Set(jsonVal))

var allModels: Array<Model>
var allSubmodelStepsVisible: Map<string, SubModelSteps> = new Map()
var respectHide = true
const urlParams = new Proxy(new URLSearchParams(window.location.search), {
    get: (searchParams, prop) => searchParams.get(prop as string),
});
const paramFilterModels = ((urlParams as any).filter as string) || ""
const paramOnlyGenerate = ((urlParams as any).gen) != undefined

const allPromptsVisible: Set<string> = new Set()
const allSamplersVisible: Set<string> = new Set()
const allCfgsVisible: Set<number> = new Set()
const allResolutionsVisible: Set<string> = new Set()

function loadVisibility() {
    const visModel = STORE_VIS_MODEL.get()
    const visSubmodel = STORE_VIS_SUBMODEL.get()
    const visSubmodelSteps = STORE_VIS_SUBMODEL_STEPS.get()

    for (const model of allModels) {
        model.visible = visModel.has(model.path)
        for (const submodel of model.submodels) {
            submodel.visible = visSubmodel.has(submodel.path)
            for (const oneSteps of submodel.submodelSteps) {
                oneSteps.visible = visSubmodelSteps.has(oneSteps.path)
                allSubmodelStepsVisible.set(oneSteps.path, oneSteps)
            }
        }
    }

    for (const prompt of STORE_VIS_PROMPT.get()) {
        allPromptsVisible.add(prompt)
    }
    for (const sampler of STORE_VIS_SAMPLER.get()) {
        allSamplersVisible.add(sampler)
    }
    for (const cfg of STORE_VIS_CFG.get()) {
        allCfgsVisible.add(cfg)
    }
    for (const res of STORE_VIS_RESOLUTION.get()) {
        allResolutionsVisible.add(res)
    }
}

function toggleVisModel(model: Model) {
    model.visible = !model.visible
    console.log(`toggle model ${model.name} to ${model.visible}`)
    if (model.visible) {
        STORE_VIS_MODEL.get().add(model.path)
    }
    else {
        STORE_VIS_MODEL.get().delete(model.path)
    }
    STORE_VIS_MODEL.save()
    renderModels()
}

function toggleVisSubmodel(submodel: SubModel) {
    submodel.visible = !submodel.visible
    console.log(`toggle submodel ${submodel.submodelStr} to ${submodel.visible}`)
    if (submodel.visible) {
        STORE_VIS_SUBMODEL.get().add(submodel.path)
    }
    else {
        STORE_VIS_SUBMODEL.get().delete(submodel.path)
    }
    STORE_VIS_SUBMODEL.save()
    renderModels()
}

function toggleVisSubmodelSteps(submodelSteps: SubModelSteps) {
    submodelSteps.visible = !submodelSteps.visible
    console.log(`toggle submodel steps to ${!submodelSteps.visible}; path = ${submodelSteps.path}`)
    if (submodelSteps.visible) {
        allSubmodelStepsVisible.set(submodelSteps.path, submodelSteps)
        STORE_VIS_SUBMODEL_STEPS.get().add(submodelSteps.path)
    }
    else {
        allSubmodelStepsVisible.delete(submodelSteps.path)
        STORE_VIS_SUBMODEL_STEPS.get().delete(submodelSteps.path)
    }
    STORE_VIS_SUBMODEL_STEPS.save()
    renderModels()
}

function toggleVisSimple<T>(value: T, visibleSet: Set<T>, store: StoredVal<Set<T>>) {
    if (visibleSet.has(value)) {
        visibleSet.delete(value)
        store.get().delete(value)
    }
    else {
        visibleSet.add(value)
        store.get().add(value)
    }
    store.save()
    renderModels()
}

function renderModels() {
    const rootElem = document.getElementById("models")!
    for (const child of Array.from(rootElem.children)) {
        if (!child.className.includes("header")) {
            rootElem.removeChild(child)
        }
    }

    for (const [modelIdx, model] of allModels.entries()) {
        if (paramFilterModels && !model.name.includes(paramFilterModels)) {
            // HACK
            continue
        }
        if (paramOnlyGenerate && !model.canGenerate) {
            continue
        }
        const modelClass = `model_${modelIdx}`
        for (const field of MODEL_FIELDS) {
            const value = model[field].toString() + ((field == "name" && model.canGenerate) ? "*" : "")
            const fieldElem = rootElem.appendChild(createElement("span", {class: field}, value))
            if (!model.visible) {
                fieldElem.classList.add(DESELECTED)
            }
            fieldElem.onclick = function(ev) { 
                toggleVisModel(model)
                return false
            }
        }

        for (const [submodelIdx, submodel] of model.submodels.entries()) {
            const submodelElems = new Array<HTMLElement>()
            if (paramOnlyGenerate && !submodel.canGenerate) {
                continue
            }

            const submodelClass = `${modelClass}_${submodelIdx}`
            const extrasString = Array.from(submodel.extras).join(" ")
            if (extrasString != "") {
                submodelElems.push(createElement("span", {class: "extras"}, extrasString))
            }

            for (const field of SUBMODEL_FIELDS) {
                // console.log(`field = ${field}`)
                const contents = submodel[field].toString() + ((field == "seed" && submodel.canGenerate) ? "*" : "")
                if (contents != "") {
                    submodelElems.push(createElement("span", {class: field}, contents))
                }
            }

            for (const elem of submodelElems) {
                if (!submodel.visible || !model.visible) {
                    elem.classList.add(DESELECTED)
                }
                rootElem.appendChild(elem)
                elem.onclick = function(ev) {
                    toggleVisSubmodel(submodel)
                    return false
                }
            }

            const stepsElem = createElement("span", {class: "submodelSteps"})
            for (const oneSteps of submodel.submodelSteps) {
                var stepsStr = oneSteps.steps.toString()
                if (oneSteps.canGenerate) {
                    stepsStr += "*"
                }
                const stepElem = stepsElem.appendChild(createElement("span", {class: "stepChoice"}, stepsStr))
                if (!oneSteps.visible || !submodel.visible || !model.visible) {
                    stepElem.classList.add(DESELECTED)
                }
                // console.log(`oneSteps.step = ${oneSteps.steps} oneSteps.path = ${oneSteps.path} oneSteps.imagesets.length = ${oneSteps.imagesets.length}`)
                stepElem.onclick = function(ev) {
                    toggleVisSubmodelSteps(oneSteps)
                    return false
                }
            }

            rootElem.append(stepsElem)
        }
    }

    renderImages(rootElem)
}

function renderImages(rootElem: HTMLElement) {
    const imagesElem = document.getElementById("images")!
    for (const child of Array.from(imagesElem.children)) {
        imagesElem.removeChild(child)
    }

    const allPrompts = new Map<string, number>()
    const allSamplers = new Map<string, number>()
    const allCfgs = new Map<number, number>()
    const allResolutions = new Map<string, number>()
    const visibleImages = new Array<Image>()

    for (const stepsPath of sort(allSubmodelStepsVisible.keys())) {
        const oneSteps = allSubmodelStepsVisible.get(stepsPath)!
        const submodel = oneSteps.submodel
        const model = submodel.model
        if (paramFilterModels && !model.name.includes(paramFilterModels)) {
            continue
        }
        if (paramOnlyGenerate && !submodel.canGenerate) {
            continue
        }
        if (!oneSteps.visible || !submodel.visible || !model.visible) {
            continue
        }

        for (const imageset of oneSteps.imagesets) {
            const prompt = imageset.prompt
            if (respectHide && imageset.hide) {
                continue
            }
            if (!allPrompts.has(imageset.prompt)) {
                allPrompts.set(imageset.prompt, 0)
            }
            allPrompts.set(imageset.prompt, allPrompts.get(imageset.prompt)! + imageset.images.length)
            if (!allPromptsVisible.has(imageset.prompt)) {
                continue
            }

            if (!allSamplers.has(imageset.samplerStr)) {
                allSamplers.set(imageset.samplerStr, 0)
            }
            allSamplers.set(imageset.samplerStr, allSamplers.get(imageset.samplerStr)! + imageset.images.length)
            if (!allSamplersVisible.has(imageset.samplerStr)) {
                continue
            }

            if (!allCfgs.has(imageset.cfg)) {
                allCfgs.set(imageset.cfg, 0)
            }
            allCfgs.set(imageset.cfg, allCfgs.get(imageset.cfg)! + imageset.images.length)
            if (!allCfgsVisible.has(imageset.cfg)) {
                continue
            }

            if (!allResolutions.has(imageset.resolution())) {
                allResolutions.set(imageset.resolution(), 0)
            }
            allResolutions.set(imageset.resolution(), allResolutions.get(imageset.resolution())! + imageset.images.length)
            if (!allResolutionsVisible.has(imageset.resolution())) {
                continue
            }

            for (const image of imageset.images) {
                visibleImages.push(image)
            }
        }
    }

    function renderChoice<T>(choice: T, visibleSet: Set<T>, store: StoredVal<Set<T>>, numAll: number, choiceStr: string = "") {
        if (!choiceStr) {
            choiceStr = "" + choice
        }
        const choiceDesc = choiceStr + ` (${numAll.toString()} images)`
        const choiceSpan = createElement("span", {class: "choice"}, choiceDesc)
        rootElem.appendChild(choiceSpan)
        choiceSpan.onclick = function(ev) {
            toggleVisSimple(choice, visibleSet, store)
            return false
        }

        if (!visibleSet.has(choice)) {
            choiceSpan.classList.add("deselected")
        }
    }

    for (const prompt of sort(allPrompts.keys())) {
        renderChoice(prompt, allPromptsVisible, STORE_VIS_PROMPT, allPrompts.get(prompt)!)
    }

    for (const sampler of sort(allSamplers.keys())) {
        renderChoice(sampler, allSamplersVisible, STORE_VIS_SAMPLER, allSamplers.get(sampler)!)
    }

    for (const cfg of sort(allCfgs.keys())) {
        renderChoice(cfg, allCfgsVisible, STORE_VIS_CFG, allCfgs.get(cfg)!, `cfg ${cfg}`)
    }

    for (const res of sort(allResolutions.keys())) {
        renderChoice(res, allResolutionsVisible, STORE_VIS_RESOLUTION, allResolutions.get(res)!)
    }

    for (const image of visibleImages) {
        const imageSrc = "/image/" + encodeURIComponent(image.path)

        const spanElem = imagesElem.appendChild(createElement("span", {class: "image"}))
        const thumbElem = spanElem.appendChild(createElement("img", {class: "thumbnail"})) as HTMLImageElement
        thumbElem.src = imageSrc

        const detailsElem = spanElem.appendChild(createElement("span", {class: "details"}))
        detailsElem.appendChild(createElement("div", {class: "attributes"}, image.path))
        const fullsizeElem = detailsElem.appendChild(createElement("img", {class: "fullsize"})) as HTMLImageElement
        fullsizeElem.src = imageSrc
    }
}

async function loadModels() {
    document.getElementById("loading")!.className = ""
    var resp = await fetch("/api/models")

    const data = await resp.text()
    if (resp.ok) {
        allModels = new Array()
        const modelsIn = JSON.parse(data)
        for (const modelIn of modelsIn) {
            const model = Model.from_json(modelIn)
            allModels.push(model)
        }
    }
    loadVisibility()
    document.getElementById("loading")!.className = "hidden"
}

loadModels().then((val2) => {
    console.log("fetched models")
    renderModels()
    document.onkeydown = (ev) => {
        if (ev.ctrlKey || ev.metaKey || ev.altKey) {
            // ignore.
        }
        else if (ev.code == "KeyH") {
            respectHide = !respectHide
            renderModels()
            return false
        }
        else if (ev.code == "KeyR") {
            loadModels().then((_val) => renderModels())
            return false
        }
        return true
    }
})
