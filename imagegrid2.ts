import { GImage, GImageSet, ColumnHeader, sort, createElement } from "./types.js"
import { buildImageSets } from "./build.js"
import { StoredVal } from "./storage.js"

// var fields = ['modelStr', 'modelName', 'modelSeed', 'modelSteps', 'prompt', 'samplerStr', 'cfg']
var fields = ['modelStr', 'modelName', 'modelSteps', 'modelSeed', 'prompt', 'samplerStr', 'cfg']

var imagesetByFilename: Map<string, GImageSet> = new Map()
var allImageSets: Map<string, GImageSet>              // image sets by key
var allImageSetKeys: Array<string>                    // sorted
var uniqueFieldValues: Map<string, Array<Object>>     // unique sorted values for each field
var fieldValueIndex: Map<String, Map<Object, number>> // index in uniqueFieldValues for each field, value


function buildHeaders(imageSetKeys: string[]): ColumnHeader[] {
    var lastHeaders = new Map<string, ColumnHeader>() // current header by field
    var allHeaders = new Array<ColumnHeader>()

    // walk through image sets in order, building the columns out
    imageSetKeys.forEach((setKey) => {
        var imgSet = allImageSets.get(setKey) as GImageSet

        for (const [idx, field] of fields.entries()) {
            var header = lastHeaders.get(field)
            if (header == null || header?.value != imgSet![field]) {
                var column = (header != null) ? header.columnEnd : 2
                header = new ColumnHeader(idx + 1, field, imgSet![field], column)
                lastHeaders.set(field, header)
                allHeaders.push(header)
            }
            header.columnEnd ++
        }
    })

    // now walk through the headers again. add classes to each of the headers such that it 
    // correctly nests within the appropriate other headers.
    var curFieldValues = new Map<String, any>()
    var curFieldColumnEnds = new Map<String, number>()
    var headersToUpdate = new Array<ColumnHeader>()
    var curColumn = 2
    for (var i = 0; i <= allHeaders.length; i ++) {
        var header = i < allHeaders.length ? allHeaders[i] : null;
        if (header == null || header.columnStart > curColumn) {
            while (headersToUpdate.length > 0) {
                const toUpdate = headersToUpdate.pop()!
                var classes = new Array<String>()
                for (const field of fields) {
                    // do not add a class to toUpdate if its end column is higher
                    // than the driving header (header)
                    if (toUpdate.columnEnd > curFieldColumnEnds.get(field)!) {
                        continue
                    }
                    var value = curFieldValues.get(field)!
                    var valueIndex = fieldValueIndex.get(field)!.get(value)!
                    classes.push(`${field}_${valueIndex}`)
                }
                toUpdate.classes = classes.join(" ")
            }
        }
        if (header != null) {
            // keep track of current value for each field, and the column that
            // that value ends on. build up a list of headers that need to be updated.
            curFieldValues.set(header.field, header.value)
            curFieldColumnEnds.set(header.field, header.columnEnd)
            curColumn = header.columnStart
            headersToUpdate.push(header)
        }
    }
    return allHeaders
}

const hiddenState = new StoredVal('hidden', new Set<String>(), 
                                    (storage) => Array.from(storage),
                                    (jsonVal) => new Set(jsonVal))
function isHidden(field: String, value: any): boolean {
    var key = `${field}/${value}`
    return hiddenState.get().has(key)
}

type Visibility = ("toggle" | "hide" | "show")
function setVisibility(field: string, value: any, visibility: Visibility): Visibility {
    var index = fieldValueIndex.get(field)?.get(value)
    if (index == undefined) {
        console.log(`can't find index for ${field} ${value}`)
        return "toggle"
    }

    var className = `${field}_${index}`
    var spanId = `choice_${className}`
    var span = document.getElementById(spanId)
    if (span != null) {
        var curHidden = isHidden(field, value)
        var newHidden: boolean

        if (visibility == "hide") {
            newHidden = true
        }
        else if (visibility == "show") {
            newHidden = false
        }
        else {
            newHidden = !curHidden
        }

        span.className = newHidden ? "" : "selected"

        for (const el of document.getElementsByClassName(className)) {
            if (newHidden) {
                el.className = el.className + " hidden"
            }
            else {
                el.className = el.className.replace(" hidden", "")
            }
        }

        const storageId = `${field}/${value}`
        if (newHidden) {
            hiddenState.get().add(storageId)
        }
        else {
            hiddenState.get().delete(storageId)
        }
        hiddenState.save()

        return newHidden ? "hide" : "show"
    }
    console.log(`can't find span ${spanId}`)
    return "toggle"
}

function onclickChoice(field: string, choice: any): any {
    var visibility = setVisibility(field, choice, "toggle")

    // if toggling a modelName, also toggle the modelStr's that are subsets of it.
    if (field == 'modelName' || field == 'modelSeed' || field == 'modelSteps') {
        var matchingModelStrs = uniqueFieldValues.get('modelStr') as Array<string>
        for (const [matchIdx, matchChoice] of matchingModelStrs.entries()) {
            if (field == 'modelName' && matchChoice.startsWith(choice as string)) {
                // modelStr that starts with this modelName should be matched.
            }
            else if (field == 'modelSeed') {
                // modelStr that has r{seed} in it should match.
                var seedStr = ` r${choice} `
                if (matchChoice.indexOf(seedStr) == -1) {
                    continue;
                }
            }
            else if (field == 'modelSteps' && matchChoice.endsWith(choice as string)) {
                // modelStr that ends with _{steps} should match
            }
            else {
                // everything else shouldn't match.
                continue;
            }

            setVisibility('modelStr', matchChoice, visibility)
        }
    }
}

function renderChoices(field: string) {
    var choices = uniqueFieldValues.get(field)!
    var chooserDiv = document.getElementById('chooser')!

    var span = document.createElement("span")
    span.className = "field"
    span.appendChild(document.createTextNode(field))
    chooserDiv.appendChild(span)

    span = document.createElement("span")
    span.className = "values"
    chooserDiv.appendChild(span)

    for (const [idx, choice] of choices.entries()) {
        var choiceSpan = document.createElement("span")
        var fieldClass = `${field}_${idx}`
        choiceSpan.className = "selected"
        choiceSpan.id = `choice_${fieldClass}`

        choiceSpan.onclick = function(this: GlobalEventHandlers, ev: MouseEvent): any {
            onclickChoice(field, choice)
        }
        choiceSpan.appendChild(document.createTextNode(choice.toString()))
        span.appendChild(choiceSpan)
    }
}

async function updateImages() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")

        allImageSets = buildImageSets(fields, filenames)
        allImageSetKeys = sort(allImageSets.keys()) as string[]
        for (const iset of allImageSets.values()) {
            for (const img of iset.images) {
                imagesetByFilename.set(img.filename, iset)
            }
        }

        // build sorted list of unique values for each field.
        // start by building a set.
        var uniqueFieldValuesSet = new Map<string, Set<any>>()
        for (const field of fields) {
            var valueSet = new Set<any>()
            uniqueFieldValuesSet.set(field, valueSet)
            for (const iset of allImageSets.values()) {
                valueSet.add(iset[field])
            }
        }

        // then convert it to a sorted array
        uniqueFieldValues = new Map<string, Array<any>>()
        for (const field of fields) {
            var val = uniqueFieldValuesSet.get(field)!
            uniqueFieldValues.set(field, sort(val))
        }

        fieldValueIndex = new Map<string, Map<Object, number>>()
        for (const field of fields) {
            var valueMap = new Map<any, number>()
            fieldValueIndex.set(field, valueMap)
            for (const [idx, value] of uniqueFieldValues.get(field)!.entries()) {
                valueMap.set(value, idx)
            }
        }
    }
    else {
        console.log("error")
    }
}

const imagesSelected = new StoredVal('images_selected', new Set<string>(), storage => Array.from(storage), jsonVal => new Set(jsonVal))
function onclickThumbnail(ev: MouseEvent, filename: string) {
    var filenamesSelected = imagesSelected.get()
    var isChecked = filenamesSelected.has(filename)
    var newChecked = !isChecked

    var imgElement = ev.target as HTMLElement
    var selectElem = imgElement.parentElement?.getElementsByClassName("image_select")?.item(0)
    if (selectElem == null) {
        console.log(`logic error: can't find image_select span for filename ${filename}`)
        return
    }

    if (newChecked) {
        selectElem.className += " checked"
        filenamesSelected.add(filename)
    }
    else {
        selectElem.className = selectElem.className.replace(" checked", "")
        filenamesSelected.delete(filename)
    }
    imagesSelected.save()
    renderCheckStats()
}

function renderCheckStats() {
    var resultsElem = document.getElementById("checked_results")
    if (resultsElem == null) {
        console.log("resultsElem not found")
        return
    }

    var html = ""

    for (const field of fields) {
        var fieldStats = new Map<any, number>()
        for (const filename of imagesSelected.get()) {
            var iset = imagesetByFilename.get(filename)
            if (iset == null) {
                console.log(`renderCheckStats: can't find imageset for filename ${filename}`)
                continue
            }
            const fieldVal = iset[field]
            var curCount = 0
            if (!fieldStats.has(fieldVal)) {
                fieldStats.set(fieldVal, 0)
            }
            else {
                curCount = fieldStats.get(fieldVal)!
            }
            fieldStats.set(fieldVal, curCount + 1)
        }
        var values = Array.from(fieldStats.keys()).sort((a, b) => {
            var aval = fieldStats.get(a)!
            var bval = fieldStats.get(b)!
            return bval - aval
        })
        html += `${field}:<ul/>\n`
        for (const [idx, value] of values.entries()) {
            html += "<li>"
            if (field == 'prompt') {
                html += `"${value}"`
            }
            else {
                html += value.toString()
            }
            html += `: ${fieldStats.get(value)}</li>\n`
        }
        html += "</ul>\n"
    }
    resultsElem.innerHTML = html
}

async function updateAndRender() {
    await updateImages()

    renderChoices('modelStr')
    renderChoices('modelName')
    renderChoices('modelSeed')
    renderChoices('modelSteps')
    renderChoices('prompt')
    renderChoices('samplerStr')
    renderChoices('cfg')

    var allHeaders = buildHeaders(allImageSetKeys)
    var grid = document.getElementById("imagegrid") as HTMLElement
    allHeaders.forEach((header) => {
        var span = createElement('span', {'class': header.classes})
        span.style.gridRow = header.row.toString()
        span.style.gridColumnStart = header.columnStart.toString()
        span.style.gridColumnEnd = header.columnEnd.toString()
        span.textContent = header.value
        grid.appendChild(span)
    })

    // generate row labels for all the seeds
    var allSeedsSet = new Set<number>()
    for (const iset of allImageSets.values()) {
        for (const img of iset.images) {
            allSeedsSet.add(img.seed)
        }
    }
    var allSeeds = sort(allSeedsSet)
    for (const [idx, seed] of allSeeds.entries()) {
        var span = createElement('span')
        span.style.gridRow = (idx + fields.length + 1).toString()
        span.style.gridColumn = "1"
        grid.appendChild(span)
    }

    // do the images!
    // var imagesHTML = ""
    var filenamesSelected = imagesSelected.get()
    for (const [isetIdx, setKey] of allImageSetKeys.entries()) {
        var iset = allImageSets.get(setKey) as GImageSet
        var column = isetIdx + 2
        var classes = fields.map((field) => {
            var val = iset[field]
            return `${field}_${fieldValueIndex.get(field)?.get(val)}`
        }).join(" ")

        for (const [imgIdx, img] of iset.images.entries()) {
            var row = imgIdx + fields.length + 1
            
            var topSpan = createElement('span', {'class': `image ${classes}`})
            topSpan.style.gridRow = row.toString()
            topSpan.style.gridColumn = column.toString()
            var selectElem = topSpan.appendChild(createElement('span', {'class': 'image_select'}, "checked"))
            if (filenamesSelected.has(img.filename)) {
                selectElem.className += " checked"
            }

            var thumbElem = topSpan.appendChild(createElement('img', {'src': img.filename, 'class': "thumbnail"}))
            thumbElem.onclick = function(this, ev) {
                onclickThumbnail(ev, img.filename)
            }

            var detailsSpan = topSpan.appendChild(createElement('span', {'class': "details"}))
            var imageElem = detailsSpan.appendChild(createElement('img', {'src': img.filename, 'class': "fullsize"}))
            var detailsGrid = detailsSpan.appendChild(createElement('div', {'class': "details_grid"}))

            var entries = {"model": iset.modelStr, "prompt": iset.prompt, 
                           "sampler": `${iset.sampler} ${iset.samplerSteps}`,
                           "CFG": iset.cfg.toString(), "seed": img.seed.toString()}
            for (const key in entries) {
                const value = entries[key]
                var keySpan = createElement('span', {'class': "detailsKey"})
                keySpan.textContent = key
                var valueSpan = createElement('span', {'class': "detailsVal"})
                valueSpan.textContent = value
                detailsGrid.appendChild(keySpan)
                detailsGrid.appendChild(valueSpan)
            }

            grid.appendChild(topSpan)
        }
    }
    renderCheckStats()

    var hidden = hiddenState.get()
    for (const hiddenStr of hidden) {
        var [field, value] = hiddenStr.split("/") as [string, any]
        if (["modelSteps", "modelSeed", "cfg"].indexOf(field) != -1) {
            value = parseInt(value)
        }
        setVisibility(field as string, value, "hide")
    }
    return null;
}

updateAndRender().then((val) => {
    console.log("done with updateList")
})
